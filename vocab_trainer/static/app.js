// Vocab Trainer — SPA Client

const API = '';
let currentSessionId = null;
let questionStartTime = null;
let currentQuestionContext = null;
let chatHistory = [];
let chatStreaming = false;
let activeAudioElements = [];  // all playing/playable audio to stop on navigation
let pendingAudioTimeout = null;
let autoCompareEnabled = localStorage.getItem('autoCompare') === 'true';
let thinkingEnabled = localStorage.getItem('llmThinking') === 'true';
let narrationQueue = null;

// ── Stick-to-bottom auto-scroll ─────────────────────────────────────────
// Uses ResizeObserver to detect content growth + wheel events to detect
// user scroll-up intent.  Smooth scrolling throughout.
const chatScroll = {
    sticky: true,           // are we "stuck" to the bottom?
    programmatic: false,    // is the current scroll animation ours?
    container: null,
    observer: null,

    init() {
        this.container = document.getElementById('chat-messages');
        if (!this.container) return;

        // ResizeObserver fires when content grows (new tokens, new messages)
        this.observer = new ResizeObserver(() => {
            if (this.sticky) this._scroll();
        });
        this.observer.observe(this.container);

        // Wheel event: upward scroll breaks stickiness immediately
        this.container.addEventListener('wheel', (e) => {
            if (e.deltaY < 0) {
                this.sticky = false;
                this._updateButton();
            }
        }, { passive: true });

        // Touch: break stickiness during touch, re-check after momentum
        this.container.addEventListener('touchstart', () => {
            this.sticky = false;
            this._updateButton();
        }, { passive: true });
        this.container.addEventListener('touchend', () => {
            // After momentum settles, check if we ended up at the bottom
            setTimeout(() => this._checkReengage(), 800);
        }, { passive: true });

        // Re-engage stickiness when user scrolls back to bottom
        this.container.addEventListener('scroll', () => {
            if (!this.programmatic) {
                this._checkReengage();
            }
        }, { passive: true });

        // scrollend cleans up programmatic flag reliably
        this.container.addEventListener('scrollend', () => {
            this.programmatic = false;
            this._checkReengage();
        });

        // Scroll-to-bottom button
        const btn = document.getElementById('scroll-to-bottom');
        if (btn) btn.addEventListener('click', () => this.scrollToBottom());
    },

    _isAtBottom() {
        const c = this.container;
        return c.scrollHeight - c.scrollTop - c.clientHeight < 5;
    },

    _checkReengage() {
        if (!this.sticky && this._isAtBottom()) {
            this.sticky = true;
        }
        this._updateButton();
    },

    _scroll() {
        if (!this.container) return;
        this.programmatic = true;
        this.container.scrollTo({
            top: this.container.scrollHeight,
            behavior: 'smooth',
        });
        // Fallback: clear programmatic flag after animation
        // (scrollend may not fire in all browsers)
        clearTimeout(this._scrollTimer);
        this._scrollTimer = setTimeout(() => { this.programmatic = false; }, 300);
    },

    _updateButton() {
        const btn = document.getElementById('scroll-to-bottom');
        if (btn) btn.classList.toggle('hidden', this.sticky);
    },

    /** Force re-engage and scroll (e.g. user sends a new message) */
    scrollToBottom() {
        this.sticky = true;
        this._scroll();
        this._updateButton();
    },
};

function setAutoCompare(enabled) {
    autoCompareEnabled = enabled;
    localStorage.setItem('autoCompare', enabled);
}

function stopAllAudio() {
    if (narrationQueue) { narrationQueue.stop(); narrationQueue = null; }
    if (pendingAudioTimeout) { clearTimeout(pendingAudioTimeout); pendingAudioTimeout = null; }
    activeAudioElements.forEach(a => { a.pause(); a.currentTime = 0; });
    activeAudioElements = [];
    // Also stop the built-in TTS player
    const tts = document.getElementById('tts-audio');
    if (tts) { tts.pause(); tts.currentTime = 0; tts.onended = null; }
}

// ── Navigation ───────────────────────────────────────────────────────────

document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
});

function switchView(view) {
    stopAllAudio();
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`.nav-btn[data-view="${view}"]`).classList.add('active');
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${view}`).classList.add('active');

    if (view === 'dashboard') refreshStats();
    if (view === 'library') refreshLibrary();
    if (view === 'settings') loadSettings();
}

// ── Dashboard ────────────────────────────────────────────────────────────

async function refreshStats() {
    try {
        const stats = await api('/api/stats');
        document.getElementById('stat-total').textContent =
            stats.total_words + ' / ' + stats.total_clusters;
        document.getElementById('stat-due').textContent = stats.words_due;
        document.getElementById('stat-reviewed').textContent = stats.words_reviewed;
        document.getElementById('stat-accuracy').textContent = stats.accuracy + '%';
        document.getElementById('stat-sessions').textContent = stats.total_sessions;
        document.getElementById('stat-active').textContent = stats.active_words;
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

document.getElementById('btn-start-session').addEventListener('click', startSession);
document.getElementById('btn-import').addEventListener('click', importVocab);
document.getElementById('btn-generate').addEventListener('click', generateQuestions);

async function importVocab() {
    const btn = document.getElementById('btn-import');
    btn.disabled = true;
    btn.textContent = 'Importing...';
    showMessage('dashboard-message', 'Importing vocabulary files...', 'info');

    try {
        const result = await api('/api/import', 'POST');
        showMessage('dashboard-message',
            `Imported ${result.words_imported} words, ${result.clusters_imported} clusters. ` +
            `Total: ${result.total_words} words, ${result.total_clusters} clusters.`,
            'success');
        refreshStats();
    } catch (e) {
        showMessage('dashboard-message', 'Import failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Import Vocabulary';
    }
}

async function generateQuestions() {
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.textContent = 'Generating...';
    showMessage('dashboard-message', 'Generating questions (this may take a minute)...', 'info');

    try {
        const result = await api('/api/generate', 'POST', { count: 10 });
        showMessage('dashboard-message',
            `Generated ${result.generated} questions. Bank size: ${result.bank_size}`,
            'success');
        refreshStats();
    } catch (e) {
        showMessage('dashboard-message', 'Generation failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Generate Questions';
    }
}

// ── Quiz Session ─────────────────────────────────────────────────────────

async function startSession() {
    await doStartSession();
}

async function doStartSession() {
    switchView('quiz');
    showQuizState('loading');

    try {
        const data = await api('/api/session/start', 'POST');
        if (data.error) {
            showQuizState('idle');
            switchView('dashboard');
            showMessage('dashboard-message', data.error, 'error');
            return;
        }
        if (data.session_complete && !data.stem) {
            showQuizState('idle');
            switchView('dashboard');
            showMessage('dashboard-message',
                'No questions available. Generate questions first, or check your LLM provider.', 'error');
            return;
        }
        currentSessionId = data.session_id;
        showQuestion(data);
    } catch (e) {
        showQuizState('idle');
        switchView('dashboard');
        showMessage('dashboard-message', 'Failed to start session: ' + e.message, 'error');
    }
}

function showQuizState(state) {
    ['idle', 'loading', 'question', 'summary'].forEach(s => {
        const el = document.getElementById(`quiz-${s}`);
        el.classList.toggle('hidden', s !== state);
    });
}

function showQuestion(data) {
    stopAllAudio();

    if (data.session_complete) {
        showSummary(data.summary || { total: 0, correct: 0, accuracy: 0 });
        return;
    }

    showQuizState('question');
    document.getElementById('answer-reveal').classList.add('hidden');

    // Reset chat and archive state
    chatHistory = [];
    currentQuestionContext = null;
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('chat-input').value = '';
    document.getElementById('archive-status').classList.add('hidden');

    // Progress (soft target)
    const progress = data.progress;
    const target = progress.target || progress.total;
    const pct = Math.min(100, (progress.current - 1) / target * 100);
    document.getElementById('progress-bar').style.width = pct + '%';
    let progressText;
    if (progress.answered >= target) {
        progressText = `${progress.answered} answered (target: ${target})`;
    } else {
        progressText = `Question ${progress.current} of ${target}`;
    }
    if (progress.answered > 0) progressText += ` | ${progress.correct}/${progress.answered} correct`;
    document.getElementById('progress-text').textContent = progressText;

    // Question type + New/Review badge
    const typeLabels = {
        fill_blank: 'Fill in the Blank',
        best_fit: 'Best Fit',
        distinction: 'Distinction',
    };
    const typeEl = document.getElementById('question-type');
    typeEl.textContent = typeLabels[data.question_type] || data.question_type;
    const badge = document.getElementById('question-badge');
    if (data.is_new) {
        badge.textContent = 'New';
        badge.className = 'question-badge new';
    } else {
        badge.textContent = 'Review';
        badge.className = 'question-badge review';
    }

    // Cluster
    document.getElementById('cluster-title').textContent =
        data.cluster_title ? `Cluster: ${data.cluster_title}` : '';

    // Stem — replace ___ with styled blank
    const stem = data.stem.replace(/___+/g, '<span class="blank">&nbsp;</span>');
    document.getElementById('question-stem').innerHTML = stem;

    // Choices
    const choicesEl = document.getElementById('choices');
    choicesEl.innerHTML = '';
    const keys = ['A', 'B', 'C', 'D'];
    data.choices.forEach((choice, i) => {
        const btn = document.createElement('button');
        btn.className = 'choice-btn';
        btn.innerHTML = `<span class="key">${keys[i]}</span><span>${choice}</span>`;
        btn.addEventListener('click', (e) => {
            // Ignore clicks that are actually text selections ending over a button
            const sel = window.getSelection();
            if (sel && sel.toString().length > 0) return;
            submitAnswer(i, data);
        });
        choicesEl.appendChild(btn);
    });

    questionStartTime = Date.now();
}

async function submitAnswer(selectedIndex, questionData) {
    // Disable all choice buttons
    const buttons = document.querySelectorAll('.choice-btn');
    buttons.forEach(btn => btn.classList.add('disabled'));

    // Highlight correct/wrong
    buttons.forEach((btn, i) => {
        if (i === questionData.correct_index) btn.classList.add('correct');
        if (i === selectedIndex && i !== questionData.correct_index) btn.classList.add('wrong');
    });

    const timeSeconds = (Date.now() - questionStartTime) / 1000;

    try {
        const result = await api('/api/session/answer', 'POST', {
            session_id: currentSessionId,
            selected_index: selectedIndex,
            time_seconds: timeSeconds,
        });

        // Show answer reveal
        const reveal = document.getElementById('answer-reveal');
        reveal.classList.remove('hidden');

        document.getElementById('result-icon').textContent =
            result.correct ? 'Correct' : 'Incorrect';
        document.getElementById('result-icon').style.color =
            result.correct ? 'var(--success)' : 'var(--error)';

        document.getElementById('explanation').textContent = result.explanation;
        document.getElementById('context-sentence').textContent = result.context_sentence;

        // Render choice details with per-word action buttons
        const detailsEl = document.getElementById('choice-details');
        detailsEl.innerHTML = '';
        const details = questionData.choice_details && questionData.choice_details.length
            ? questionData.choice_details
            : questionData.choices.map(c => ({ word: c, meaning: '', distinction: '' }));

        const heading = document.createElement('h4');
        heading.textContent = 'All choices';
        detailsEl.appendChild(heading);

        details.forEach((d, i) => {
            const item = document.createElement('div');
            const isCorrect = i === questionData.correct_index;
            item.className = 'choice-detail' + (isCorrect ? ' correct' : '');

            let html = `<div class="choice-detail-text"><strong>${d.word}</strong>`;
            if (d.meaning) html += ` — ${d.meaning}`;
            if (d.why) html += `<span class="choice-why">${d.why}</span>`;
            else if (d.distinction) html += `<span class="distinction">${d.distinction}</span>`;
            html += '</div>';
            html += `<div class="choice-detail-actions">` +
                `<button class="word-action-btn" data-word="${d.word}" data-action="context">Context</button>` +
                `<button class="word-action-btn" data-word="${d.word}" data-action="etymology">Etymology</button>` +
                `</div>`;

            item.innerHTML = html;
            detailsEl.appendChild(item);
        });

        // Show archive decision with override
        const archiveEl = document.getElementById('archive-status');
        const archiveText = document.getElementById('archive-text');
        const archiveToggle = document.getElementById('archive-toggle');
        if (result.archive && result.archive.question_id) {
            archiveEl.classList.remove('hidden', 'archived', 'kept');
            if (result.archive.archived) {
                archiveEl.classList.add('archived');
                archiveText.innerHTML = `<strong>Archived</strong> — ${result.archive.reason}`;
                archiveToggle.textContent = 'Keep in rotation';
                archiveToggle.onclick = () => toggleArchive(result.archive.question_id, false);
            } else {
                archiveEl.classList.add('kept');
                const iv = result.archive.interval_days || 0;
                const th = result.archive.archive_threshold || 21;
                const progress = iv > 0
                    ? ` — interval ${Math.round(iv)} of ${th} days`
                    : '';
                archiveText.innerHTML = `<strong>In rotation</strong>${progress}`;
                archiveToggle.textContent = 'Archive';
                archiveToggle.onclick = () => toggleArchive(result.archive.question_id, true);
            }
        } else {
            archiveEl.classList.add('hidden');
        }

        // Store context for chat
        currentQuestionContext = {
            question_type: questionData.question_type,
            stem: questionData.stem,
            choices: questionData.choices,
            correct_index: questionData.correct_index,
            correct_word: questionData.correct_word,
            explanation: result.explanation,
            context_sentence: result.context_sentence,
            cluster_title: questionData.cluster_title,
            choice_details: questionData.choice_details || [],
            selected_index: selectedIndex,
            was_correct: result.correct,
        };

        // Audio: narrate the completed sentence only
        stopAllAudio();
        const audio = document.getElementById('tts-audio');
        if (result.context_audio_hash) {
            audio.src = `/api/audio/${result.context_audio_hash}.mp3`;
            audio.hidden = false;
            audio.play().catch(() => {});
        }

        // Update progress text (soft target)
        const sp = result.session_progress;
        const spTarget = sp.target || (sp.answered + sp.remaining);
        let spText;
        if (sp.answered >= spTarget) {
            spText = `${sp.answered} answered (target: ${spTarget}) | ${sp.correct}/${sp.answered} correct`;
        } else {
            spText = `${sp.correct}/${sp.answered} correct | ${sp.remaining} remaining`;
        }
        document.getElementById('progress-text').textContent = spText;
        const spPct = Math.min(100, sp.answered / spTarget * 100);
        document.getElementById('progress-bar').style.width = spPct + '%';

        // Show/hide finish button once target is reached
        const finishBtn = document.getElementById('btn-finish');
        if (sp.answered >= spTarget && !result.session_complete) {
            finishBtn.classList.remove('hidden');
        } else {
            finishBtn.classList.add('hidden');
        }

        // Finish button handler
        finishBtn.onclick = async () => {
            stopAllAudio();
            try {
                const finishResult = await api('/api/session/finish', 'POST', {
                    session_id: currentSessionId,
                });
                showSummary(finishResult.summary);
            } catch (e) {
                console.error('Finish session failed:', e);
            }
        };

        // Next button
        const nextBtn = document.getElementById('btn-next');
        nextBtn.onclick = async () => {
            stopAllAudio();
            if (result.session_complete) {
                showSummary(result.summary);
            } else {
                showQuizState('loading');
                document.getElementById('quiz-loading-text').textContent = 'Generating question...';
                const nextQ = await api('/api/session/next', 'POST', {
                    session_id: currentSessionId,
                });
                document.getElementById('quiz-loading-text').textContent = 'Loading question...';
                showQuestion(nextQ);
            }
        };
    } catch (e) {
        console.error('Answer submission failed:', e);
    }
}

async function toggleArchive(questionId, archived) {
    try {
        await api(`/api/question/${questionId}/archive`, 'POST', { archived });
        const archiveEl = document.getElementById('archive-status');
        const archiveText = document.getElementById('archive-text');
        const archiveToggle = document.getElementById('archive-toggle');
        archiveEl.classList.remove('archived', 'kept');
        if (archived) {
            archiveEl.classList.add('archived');
            archiveText.innerHTML = '<strong>Archived</strong> — manually archived';
            archiveToggle.textContent = 'Keep in rotation';
            archiveToggle.onclick = () => toggleArchive(questionId, false);
        } else {
            archiveEl.classList.add('kept');
            archiveText.innerHTML = '<strong>In rotation</strong> — restored';
            archiveToggle.textContent = 'Archive';
            archiveToggle.onclick = () => toggleArchive(questionId, true);
        }
    } catch (e) {
        console.error('Archive toggle failed:', e);
    }
}

function showSummary(summary) {
    showQuizState('summary');
    document.getElementById('summary-total').textContent = summary.total;
    document.getElementById('summary-correct').textContent = summary.correct;
    document.getElementById('summary-accuracy').textContent = summary.accuracy + '%';
    const breakdown = document.getElementById('summary-breakdown');
    const rc = summary.review_count || 0;
    const nc = summary.new_count || 0;
    if (rc > 0 || nc > 0) {
        breakdown.textContent = rc + ' review + ' + nc + ' new';
        breakdown.classList.remove('hidden');
    } else {
        breakdown.classList.add('hidden');
    }
    currentSessionId = null;
}

document.getElementById('btn-new-session').addEventListener('click', startSession);
document.getElementById('btn-back-dashboard').addEventListener('click', () => {
    switchView('dashboard');
});

// ── Keyboard shortcuts ───────────────────────────────────────────────────

document.addEventListener('keydown', (e) => {
    // Don't intercept when typing in the chat input
    if (document.activeElement && document.activeElement.id === 'chat-input') return;

    const quizQuestion = document.getElementById('quiz-question');
    if (quizQuestion.classList.contains('hidden')) return;

    const reveal = document.getElementById('answer-reveal');
    if (!reveal.classList.contains('hidden')) {
        // Answer revealed — Enter/Space for next
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            document.getElementById('btn-next').click();
        }
        return;
    }

    // Choice selection via A-D or 1-4
    const keyMap = { a: 0, b: 1, c: 2, d: 3, '1': 0, '2': 1, '3': 2, '4': 3 };
    const index = keyMap[e.key.toLowerCase()];
    if (index !== undefined) {
        const buttons = document.querySelectorAll('.choice-btn');
        if (buttons[index] && !buttons[index].classList.contains('disabled')) {
            buttons[index].click();
        }
    }
});

// ── Word Chat (LLM tutor) ───────────────────────────────────────────────

const CHAT_PROMPTS = {
    context_simple:
        'Show me 3 simple, everyday sentences using "{word}". Keep the vocabulary and situations accessible.',
    context_intermediate:
        'Show me 3 sentences using "{word}" in professional or academic contexts.',
    context_advanced:
        'Show me 3 sentences using "{word}" in literary or sophisticated contexts. Show the word at its most expressive.',
    etymology:
        'Explain the etymology and deeper meaning of "{word}". Include its roots (Latin, Greek, etc.), how the meaning evolved, and any interesting historical context.',
    compare_correct:
        'Compare all four choices. For each word, explain what it means, when you\'d use it vs the others, and give a couple of example sentences.',
    compare_wrong:
        'Compare all four choices. Start by contrasting "{chosen}" with "{correct}" — why doesn\'t my choice work here and what makes the correct answer fit? Then cover the other two.',
};

let selectedComplexity = localStorage.getItem('contextLevel') || 'simple';

// Complexity selector — restore active state and persist choice
document.querySelectorAll('.complexity-btn').forEach(btn => {
    if (btn.dataset.level === selectedComplexity) {
        document.querySelectorAll('.complexity-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    }
    btn.addEventListener('click', () => {
        document.querySelectorAll('.complexity-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedComplexity = btn.dataset.level;
        localStorage.setItem('contextLevel', selectedComplexity);
    });
});

// Global chat buttons (Compare all)
document.querySelectorAll('.chat-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (chatStreaming || !currentQuestionContext) return;
        const action = btn.dataset.action;
        if (action === 'compare') {
            const ctx = currentQuestionContext;
            const chosen = ctx.choices[ctx.selected_index];
            let message;
            if (ctx.was_correct) {
                message = CHAT_PROMPTS.compare_correct;
            } else {
                message = CHAT_PROMPTS.compare_wrong
                    .replace('{chosen}', chosen)
                    .replace('{correct}', ctx.correct_word);
            }
            sendChatMessage(message);
        }
    });
});

// Auto-compare toggle
const autoCompareCheckbox = document.getElementById('auto-compare-toggle');
autoCompareCheckbox.checked = autoCompareEnabled;
autoCompareCheckbox.addEventListener('change', () => {
    setAutoCompare(autoCompareCheckbox.checked);
});

// Thinking toggle
const thinkingCheckbox = document.getElementById('thinking-toggle');
thinkingCheckbox.checked = thinkingEnabled;
thinkingCheckbox.addEventListener('change', () => {
    thinkingEnabled = thinkingCheckbox.checked;
    localStorage.setItem('llmThinking', thinkingEnabled);
    api('/api/settings', 'PUT', { llm_thinking: thinkingEnabled }).catch(() => {});
});

// Per-word action buttons (delegated from choice-details container)
document.getElementById('choice-details').addEventListener('click', (e) => {
    const btn = e.target.closest('.word-action-btn');
    if (!btn || chatStreaming || !currentQuestionContext) return;
    const word = btn.dataset.word;
    const action = btn.dataset.action;
    let message;
    if (action === 'context') {
        const key = 'context_' + selectedComplexity;
        message = CHAT_PROMPTS[key].replace('{word}', word);
    } else if (action === 'etymology') {
        message = CHAT_PROMPTS.etymology.replace('{word}', word);
    }
    if (message) sendChatMessage(message);
});

// Text input
document.getElementById('chat-send').addEventListener('click', () => {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (msg && !chatStreaming && currentQuestionContext) {
        input.value = '';
        sendChatMessage(msg);
    }
});

document.getElementById('chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        document.getElementById('chat-send').click();
    }
});

function appendChatMessage(role, text) {
    const el = document.createElement('div');
    el.className = `chat-msg ${role}`;
    if (role === 'assistant') {
        el.innerHTML = simpleMarkdown(text);
    } else {
        el.textContent = text;
    }
    const container = document.getElementById('chat-messages');
    container.appendChild(el);
    // ResizeObserver handles scrolling; no manual call needed
    return el;
}

function addNarrateButton(msgEl, rawText) {
    const bar = document.createElement('div');
    bar.className = 'chat-msg-actions';
    const btn = document.createElement('button');
    btn.className = 'narrate-btn';
    btn.textContent = '\u25B6 Narrate';
    btn.onclick = () => narrateText(btn, rawText);
    bar.appendChild(btn);
    msgEl.appendChild(bar);
}

async function narrateText(btn, text) {
    stopAllAudio();
    btn.disabled = true;
    btn.textContent = '\u25B6 Generating...';
    try {
        const result = await api('/api/tts/generate', 'POST', { text });
        if (result.audio_hash) {
            const audio = new Audio(`/api/audio/${result.audio_hash}.mp3`);
            activeAudioElements.push(audio);
            btn.textContent = '\u25A0 Stop';
            btn.disabled = false;
            audio.onended = () => { btn.textContent = '\u25B6 Narrate'; };
            btn.onclick = () => {
                if (audio.paused) {
                    audio.play();
                    btn.textContent = '\u25A0 Stop';
                } else {
                    audio.pause();
                    audio.currentTime = 0;
                    btn.textContent = '\u25B6 Narrate';
                }
            };
            audio.play();
        }
    } catch (e) {
        btn.textContent = '\u25B6 Narrate';
        btn.disabled = false;
        console.error('Narration failed:', e);
    }
}

class NarrationQueue {
    constructor(previousQueue) {
        this.buffer = '';
        this.sentences = [];
        this.currentIndex = 0;
        this.playing = false;
        this.stopped = false;
        this.flushed = false;
        this.started = false;
        this.previousQueue = previousQueue || null;
        this.onDone = null;
    }

    feedToken(token) {
        if (this.stopped) return;
        this.buffer += token;
        this._extractSentences();
    }

    _extractSentences() {
        while (true) {
            // Paragraph break
            const paraIdx = this.buffer.indexOf('\n\n');
            if (paraIdx > 0) {
                const sentence = this.buffer.slice(0, paraIdx).trim();
                this.buffer = this.buffer.slice(paraIdx + 2);
                if (sentence) this._enqueueSentence(sentence);
                continue;
            }
            // Single newline — catches line-based output (lists, tables)
            if (this.buffer.length > 20) {
                const nlIdx = this.buffer.indexOf('\n');
                if (nlIdx > 0) {
                    const sentence = this.buffer.slice(0, nlIdx).trim();
                    this.buffer = this.buffer.slice(nlIdx + 1);
                    if (sentence) this._enqueueSentence(sentence);
                    continue;
                }
            }
            // Sentence-ending punctuation followed by whitespace
            if (this.buffer.length > 20) {
                const match = this.buffer.match(/([.!?])(\s)/);
                if (match) {
                    const endIdx = match.index + 1;
                    const sentence = this.buffer.slice(0, endIdx).trim();
                    this.buffer = this.buffer.slice(endIdx).trimStart();
                    if (sentence) this._enqueueSentence(sentence);
                    continue;
                }
            }
            break;
        }
    }

    flush() {
        if (this.stopped) return;
        const remaining = this.buffer.trim();
        this.buffer = '';
        if (remaining) this._enqueueSentence(remaining);
        this.flushed = true;
        this._tryPlay();
    }

    stop() {
        this.stopped = true;
        this.sentences.forEach(s => {
            if (s.audio) { s.audio.pause(); s.audio.currentTime = 0; }
        });
        if (this.onDone) { this.onDone(); this.onDone = null; }
    }

    _enqueueSentence(text) {
        const clean = text
            .replace(/\*\*(.+?)\*\*/g, '$1')
            .replace(/\*(.+?)\*/g, '$1')
            .replace(/`(.+?)`/g, '$1')
            .replace(/^#+\s*/gm, '')
            .replace(/^\s*[-*]\s+/gm, '')     // list markers
            .replace(/^\s*\d+\.\s+/gm, '')    // numbered lists
            .replace(/\|/g, ', ')              // table pipes
            .replace(/^[\s:_-]+$/gm, '')       // table dividers / horizontal rules
            .replace(/[*_]/g, '')              // remaining markdown emphasis chars
            .replace(/\n/g, ' ')
            .replace(/\s{2,}/g, ' ')
            .trim();
        if (!clean || clean.length < 3) return;
        const index = this.sentences.length;
        this.sentences.push({ text: clean, audio: null, ready: false, failed: false });
        this._generateAudio(index, clean);
    }

    async _generateAudio(index, text) {
        if (this.stopped) return;
        try {
            const result = await api('/api/tts/generate', 'POST', { text });
            if (this.stopped) return;
            if (result.audio_hash) {
                const audio = new Audio(`/api/audio/${result.audio_hash}.mp3`);
                activeAudioElements.push(audio);
                this.sentences[index].audio = audio;
                this.sentences[index].ready = true;
                audio.onended = () => {
                    this.playing = false;
                    this.currentIndex++;
                    this._tryPlay();
                };
                this._tryPlay();
            } else {
                this.sentences[index].failed = true;
                if (index === this.currentIndex) this._tryPlay();
            }
        } catch (e) {
            if (this.stopped) return;
            console.error('Narration TTS failed:', e);
            this.sentences[index].failed = true;
            if (index === this.currentIndex) this._tryPlay();
        }
    }

    _tryPlay() {
        if (this.stopped || this.playing) return;
        while (this.currentIndex < this.sentences.length && this.sentences[this.currentIndex].failed) {
            this.currentIndex++;
        }
        if (this.currentIndex >= this.sentences.length) {
            if (this.flushed && this.onDone) { this.onDone(); this.onDone = null; }
            return;
        }
        const sentence = this.sentences[this.currentIndex];
        if (!sentence.ready || !sentence.audio) return;
        if (!this.started) {
            this.started = true;
            // Stop previous narration and other audio, but not this queue
            if (this.previousQueue) { this.previousQueue.stop(); this.previousQueue = null; }
            if (pendingAudioTimeout) { clearTimeout(pendingAudioTimeout); pendingAudioTimeout = null; }
            activeAudioElements.forEach(a => { a.pause(); a.currentTime = 0; });
            activeAudioElements = [];
            const tts = document.getElementById('tts-audio');
            if (tts) { tts.pause(); tts.currentTime = 0; tts.onended = null; }
        }
        this.playing = true;
        sentence.audio.play().catch(() => {
            this.playing = false;
            this.currentIndex++;
            this._tryPlay();
        });
    }
}

function simpleMarkdown(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`(.+?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

async function sendChatMessage(message) {
    if (chatStreaming) return;
    chatStreaming = true;
    // Disable buttons while streaming
    document.querySelectorAll('.chat-btn, .chat-send-btn, .word-action-btn').forEach(b => b.disabled = true);

    // Set up auto-narration if enabled
    const autoNarrating = autoCompareEnabled;
    if (autoNarrating) {
        narrationQueue = new NarrationQueue(narrationQueue);
    }

    appendChatMessage('user', message);
    chatScroll.scrollToBottom();  // Force re-engage on new prompt
    const assistantEl = appendChatMessage('assistant', '');
    assistantEl.classList.add('streaming');

    // The backend pre-fills "**" after "Tutor:" on the first message to
    // force a substantive bold-word opening.  The model continues from
    // that prefix, so we prepend it here to keep the markdown valid.
    const prefill = chatHistory.length === 0 ? '**' : '';
    let fullResponse = prefill;

    try {
        const resp = await fetch(API + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                context: currentQuestionContext,
                history: chatHistory,
                thinking: thinkingEnabled,
            }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let thinkingIndicator = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });

            for (const line of chunk.split('\n')) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.thinking === true) {
                        thinkingIndicator = document.createElement('span');
                        thinkingIndicator.className = 'chat-thinking';
                        thinkingIndicator.textContent = 'Thinking...';
                        assistantEl.appendChild(thinkingIndicator);
                    }
                    if (data.thinking === false && thinkingIndicator) {
                        thinkingIndicator.remove();
                        thinkingIndicator = null;
                    }
                    if (data.token) {
                        if (thinkingIndicator) {
                            thinkingIndicator.remove();
                            thinkingIndicator = null;
                        }
                        fullResponse += data.token;
                        if (autoNarrating && narrationQueue) narrationQueue.feedToken(data.token);
                        assistantEl.innerHTML = simpleMarkdown(fullResponse);
                    }
                    if (data.error) {
                        if (thinkingIndicator) {
                            thinkingIndicator.remove();
                            thinkingIndicator = null;
                        }
                        fullResponse += `\n[Error: ${data.error}]`;
                        assistantEl.innerHTML = simpleMarkdown(fullResponse);
                    }
                } catch (e) {
                    // Ignore malformed SSE lines
                }
            }
        }

        if (autoNarrating && narrationQueue) narrationQueue.flush();
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: fullResponse });

        if (fullResponse) {
            if (autoNarrating && narrationQueue && !narrationQueue.stopped) {
                // Show stop button — narration is already playing
                const bar = document.createElement('div');
                bar.className = 'chat-msg-actions';
                const btn = document.createElement('button');
                btn.className = 'narrate-btn';
                btn.textContent = '\u25A0 Stop';
                const queue = narrationQueue;
                const switchToNarrate = () => {
                    btn.textContent = '\u25B6 Narrate';
                    btn.disabled = false;
                    btn.onclick = () => narrateText(btn, fullResponse);
                };
                btn.onclick = () => {
                    queue.stop();
                    switchToNarrate();
                };
                queue.onDone = switchToNarrate;
                bar.appendChild(btn);
                assistantEl.appendChild(bar);
            } else {
                addNarrateButton(assistantEl, fullResponse);
            }
            // ResizeObserver handles scroll; button append triggers it
        }
    } catch (e) {
        assistantEl.innerHTML = simpleMarkdown(`[Connection error: ${e.message}]`);
    } finally {
        assistantEl.classList.remove('streaming');
        chatStreaming = false;
        document.querySelectorAll('.chat-btn, .chat-send-btn, .word-action-btn').forEach(b => b.disabled = false);
    }
}

// ── Library ──────────────────────────────────────────────────────────────

document.querySelectorAll('.library-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.library-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.dataset.tab;
        document.getElementById('library-active').classList.toggle('hidden', tab !== 'active');
        document.getElementById('library-archived').classList.toggle('hidden', tab !== 'archived');
    });
});

async function refreshLibrary() {
    try {
        const [active, archived] = await Promise.all([
            api('/api/questions/active'),
            api('/api/questions/archived'),
        ]);
        renderLibraryPanel('library-active', active, false);
        renderLibraryPanel('library-archived', archived, true);
    } catch (e) {
        console.error('Failed to load library:', e);
        document.getElementById('library-active').innerHTML =
            `<p class="library-empty">Failed to load: ${e.message}</p>`;
    }
}

function renderLibraryPanel(containerId, questions, isArchived) {
    const container = document.getElementById(containerId);
    if (questions.length === 0) {
        container.innerHTML = `<p class="library-empty">${isArchived ? 'No archived questions.' : 'No active questions yet. Start a quiz session to begin.'}</p>`;
        return;
    }

    // Group by word
    const byWord = new Map();
    for (const q of questions) {
        const key = q.target_word.toLowerCase();
        if (!byWord.has(key)) byWord.set(key, []);
        byWord.get(key).push(q);
    }

    container.innerHTML = '';
    for (const [, wordQuestions] of byWord) {
        const first = wordQuestions[0];
        const totalShown = wordQuestions.reduce((s, q) => s + (q.times_shown || 0), 0);
        const totalCorrect = wordQuestions.reduce((s, q) => s + (q.times_correct || 0), 0);

        const item = document.createElement('div');
        item.className = 'question-list-item';

        // SRS status
        let srsText = '';
        if (first.next_review) {
            const now = new Date();
            const due = new Date(first.next_review);
            const diffMs = due - now;
            if (diffMs <= 0) {
                srsText = 'due now';
            } else {
                const diffDays = Math.ceil(diffMs / 86400000);
                srsText = diffDays === 1 ? 'due in 1d' : `due in ${diffDays}d`;
            }
        }

        let html = `<div class="qli-header">`;
        html += `<div class="qli-word-row">`;
        html += `<strong class="qli-word">${first.target_word}</strong>`;
        if (first.cluster_title) html += `<span class="qli-cluster">${first.cluster_title}</span>`;
        html += `</div>`;
        html += `<div class="qli-stats">`;
        html += `<span>shown ${totalShown}x, ${totalCorrect}/${totalShown} correct</span>`;
        if (srsText) html += `<span class="qli-srs ${srsText === 'due now' ? 'due-now' : ''}">${srsText}</span>`;
        html += `</div>`;
        html += `</div>`;

        html += `<div class="qli-actions">`;
        if (isArchived) {
            html += `<button class="qli-btn" data-action="restore" data-ids='${JSON.stringify(wordQuestions.map(q => q.id))}'>Restore</button>`;
        } else {
            html += `<button class="qli-btn" data-action="archive" data-ids='${JSON.stringify(wordQuestions.map(q => q.id))}'>Archive</button>`;
        }
        html += `<button class="qli-btn" data-action="reset" data-word="${first.target_word}">Reset Due</button>`;
        html += `</div>`;

        // Expandable stems
        html += `<div class="qli-stems hidden">`;
        for (const q of wordQuestions) {
            html += `<div class="qli-stem">${q.stem}</div>`;
        }
        html += `</div>`;

        item.innerHTML = html;

        // Click to expand
        item.querySelector('.qli-header').addEventListener('click', () => {
            item.classList.toggle('expanded');
            item.querySelector('.qli-stems').classList.toggle('hidden');
        });

        container.appendChild(item);
    }

    // Action handlers (delegated)
    container.addEventListener('click', async (e) => {
        const btn = e.target.closest('.qli-btn');
        if (!btn) return;
        e.stopPropagation();
        const action = btn.dataset.action;

        if (action === 'archive' || action === 'restore') {
            const ids = JSON.parse(btn.dataset.ids);
            const archived = action === 'archive';
            btn.disabled = true;
            btn.textContent = archived ? 'Archiving...' : 'Restoring...';
            for (const id of ids) {
                await api(`/api/question/${id}/archive`, 'POST', { archived });
            }
            await refreshLibrary();
        } else if (action === 'reset') {
            const word = btn.dataset.word;
            btn.disabled = true;
            btn.textContent = 'Resetting...';
            await api('/api/questions/reset-due', 'POST', { word });
            await refreshLibrary();
        }
    });
}

// ── Settings ─────────────────────────────────────────────────────────────

const TTS_VOICES = {
    'edge-tts': [
        { id: 'en-US-GuyNeural', label: 'Guy (Male, US)' },
        { id: 'en-US-AriaNeural', label: 'Aria (Female, US)' },
        { id: 'en-US-JennyNeural', label: 'Jenny (Female, US)' },
        { id: 'en-US-AndrewNeural', label: 'Andrew (Male, US)' },
        { id: 'en-US-AvaNeural', label: 'Ava (Female, US)' },
        { id: 'en-GB-RyanNeural', label: 'Ryan (Male, British)' },
        { id: 'en-GB-SoniaNeural', label: 'Sonia (Female, British)' },
    ],
    'elevenlabs': [
        { id: '21m00Tcm4TlvDq8ikWAM', label: 'Rachel' },
        { id: 'pNInz6obpgDQGcFmaJgB', label: 'Adam' },
        { id: 'nPczCjzI2devNBz1zQrb', label: 'Brian' },
        { id: 'TxGEqnHWrfWFTfGW9XjX', label: 'Josh' },
        { id: 'onwK4e9ZLuTAKqWW03F9', label: 'Daniel' },
        { id: 'Xb7hH8MSUJpSbSDYk0k2', label: 'Alice' },
        { id: 'XrExE9yKIg1WjnnlVkGX', label: 'Matilda' },
    ],
    'piper': [
        { id: 'en_US-lessac-medium', label: 'lessac-medium' },
        { id: 'en_US-lessac-high', label: 'lessac-high' },
        { id: 'en_US-ryan-medium', label: 'ryan-medium' },
        { id: 'en_US-ryan-high', label: 'ryan-high' },
        { id: 'en_US-amy-medium', label: 'amy-medium' },
        { id: 'en_US-cori-medium', label: 'cori-medium' },
    ],
};

const TTS_HINTS = {
    'edge-tts': 'Free Microsoft Edge voices. Full list: edge-tts --list-voices',
    'elevenlabs': 'Requires ELEVEN_LABS_API_KEY env var. Voice IDs from elevenlabs.io',
    'piper': 'Offline TTS. Model names from github.com/rhasspy/piper',
};

function populateVoiceDropdown(provider, currentVoiceId) {
    const select = document.getElementById('set-tts-voice-select');
    const custom = document.getElementById('set-tts-voice-custom');
    const hint = document.getElementById('tts-voice-hint');
    const modelGroup = document.getElementById('elevenlabs-model-group');

    const voices = TTS_VOICES[provider] || [];
    select.innerHTML = '';
    voices.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v.id;
        opt.textContent = v.label;
        select.appendChild(opt);
    });
    // Add "Other" option
    const otherOpt = document.createElement('option');
    otherOpt.value = '__other__';
    otherOpt.textContent = 'Other (custom ID)';
    select.appendChild(otherOpt);

    // Select current voice or fall back to "Other"
    const match = voices.find(v => v.id === currentVoiceId);
    if (match) {
        select.value = match.id;
        custom.classList.add('hidden');
    } else {
        select.value = '__other__';
        custom.value = currentVoiceId || '';
        custom.classList.remove('hidden');
    }

    hint.textContent = TTS_HINTS[provider] || '';
    modelGroup.classList.toggle('hidden', provider !== 'elevenlabs');
}

function getSelectedVoice() {
    const select = document.getElementById('set-tts-voice-select');
    if (select.value === '__other__') {
        return document.getElementById('set-tts-voice-custom').value.trim() || select.options[0].value;
    }
    return select.value;
}

async function loadSettings() {
    try {
        const s = await api('/api/settings');
        document.getElementById('set-llm').value = s.llm_provider;
        document.getElementById('set-llm-model').value = s.llm_model;
        document.getElementById('set-tts').value = s.tts_provider;
        populateVoiceDropdown(s.tts_provider, s.tts_voice);
        document.getElementById('set-elevenlabs-model').value = s.elevenlabs_model || 'eleven_flash_v2_5';
        document.getElementById('set-session-size').value = s.session_size;
        document.getElementById('set-session-size-val').textContent = s.session_size;
        document.getElementById('set-min-ready').value = s.min_ready_questions;
        document.getElementById('set-min-ready-val').textContent = s.min_ready_questions;
        document.getElementById('set-archive-interval').value = s.archive_interval_days;
        document.getElementById('set-archive-interval-val').textContent = s.archive_interval_days;
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

document.getElementById('set-session-size').addEventListener('input', (e) => {
    document.getElementById('set-session-size-val').textContent = e.target.value;
});

document.getElementById('set-min-ready').addEventListener('input', (e) => {
    document.getElementById('set-min-ready-val').textContent = e.target.value;
});

document.getElementById('set-archive-interval').addEventListener('input', (e) => {
    document.getElementById('set-archive-interval-val').textContent = e.target.value;
});

document.getElementById('set-tts').addEventListener('change', (e) => {
    const provider = e.target.value;
    const voices = TTS_VOICES[provider] || [];
    const defaultVoice = voices.length > 0 ? voices[0].id : '';
    populateVoiceDropdown(provider, defaultVoice);
});

document.getElementById('set-tts-voice-select').addEventListener('change', (e) => {
    const custom = document.getElementById('set-tts-voice-custom');
    custom.classList.toggle('hidden', e.target.value !== '__other__');
    if (e.target.value === '__other__') custom.focus();
});

document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
        await api('/api/settings', 'PUT', {
            llm_provider: document.getElementById('set-llm').value,
            llm_model: document.getElementById('set-llm-model').value,
            tts_provider: document.getElementById('set-tts').value,
            tts_voice: getSelectedVoice(),
            elevenlabs_model: document.getElementById('set-elevenlabs-model').value,
            session_size: parseInt(document.getElementById('set-session-size').value),
            min_ready_questions: parseInt(document.getElementById('set-min-ready').value),
            archive_interval_days: parseInt(document.getElementById('set-archive-interval').value),
        });
        showMessage('settings-message', 'Settings saved.', 'success');
    } catch (e) {
        showMessage('settings-message', 'Failed to save: ' + e.message, 'error');
    }
});

// ── API Helper ───────────────────────────────────────────────────────────

async function api(path, method = 'GET', body = null) {
    const opts = { method, headers: {} };
    if (body) {
        opts.headers['Content-Type'] = 'application/json';
        opts.body = JSON.stringify(body);
    }
    const resp = await fetch(API + path, opts);
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`${resp.status}: ${text}`);
    }
    return resp.json();
}

function showMessage(elementId, text, type) {
    const el = document.getElementById(elementId);
    el.textContent = text;
    el.className = `message ${type}`;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 8000);
}

// ── Init ─────────────────────────────────────────────────────────────────

chatScroll.init();
refreshStats();

// Sync thinking toggle from backend settings
api('/api/settings').then(s => {
    if (typeof s.llm_thinking === 'boolean') {
        thinkingEnabled = s.llm_thinking;
        localStorage.setItem('llmThinking', thinkingEnabled);
        document.getElementById('thinking-toggle').checked = thinkingEnabled;
    }
}).catch(() => {});
