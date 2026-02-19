// Vocab Trainer — SPA Client

const API = '';
let currentSessionId = null;
let questionStartTime = null;
let currentQuestionContext = null;
let chatHistory = [];
let chatStreaming = false;
let activeAudioElements = [];  // all playing/playable audio to stop on navigation

function stopAllAudio() {
    activeAudioElements.forEach(a => { a.pause(); a.currentTime = 0; });
    activeAudioElements = [];
    // Also stop the built-in TTS player
    const tts = document.getElementById('tts-audio');
    if (tts) { tts.pause(); tts.currentTime = 0; }
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
    if (view === 'settings') loadSettings();
}

// ── Dashboard ────────────────────────────────────────────────────────────

async function refreshStats() {
    try {
        const stats = await api('/api/stats');
        document.getElementById('stat-total').textContent = stats.total_words;
        document.getElementById('stat-reviewed').textContent = stats.words_reviewed;
        document.getElementById('stat-due').textContent = stats.words_due;
        document.getElementById('stat-accuracy').textContent = stats.accuracy + '%';
        document.getElementById('stat-sessions').textContent = stats.total_sessions;
        document.getElementById('stat-active').textContent =
            stats.active_words + ' / ' + stats.max_active_words;
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

    // Progress
    const progress = data.progress;
    const pct = ((progress.current - 1) / progress.total * 100);
    document.getElementById('progress-bar').style.width = pct + '%';
    document.getElementById('progress-text').textContent =
        `Question ${progress.current} of ${progress.total}` +
        (progress.answered > 0 ? ` | ${progress.correct}/${progress.answered} correct` : '');

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
        btn.addEventListener('click', () => submitAnswer(i, data));
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
        };

        // Audio
        if (result.audio_hash) {
            const audio = document.getElementById('tts-audio');
            audio.src = `/api/audio/${result.audio_hash}.mp3`;
            audio.hidden = false;
            audio.play().catch(() => {});
        }

        // Update progress text
        const sp = result.session_progress;
        document.getElementById('progress-text').textContent =
            `${sp.correct}/${sp.answered} correct | ${sp.remaining} remaining`;

        // Next button
        const nextBtn = document.getElementById('btn-next');
        nextBtn.onclick = async () => {
            if (result.session_complete) {
                showSummary(result.summary);
            } else {
                showQuizState('loading');
                const nextQ = await api('/api/session/next', 'POST', {
                    session_id: currentSessionId,
                });
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
    compare:
        'Compare all four choices ({choices}) in detail. For each, explain exactly when you\'d use it vs the others, with a concrete example sentence highlighting the distinction.',
};

let selectedComplexity = 'simple';

// Complexity selector
document.querySelectorAll('.complexity-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.complexity-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        selectedComplexity = btn.dataset.level;
    });
});

// Global chat buttons (Compare all)
document.querySelectorAll('.chat-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (chatStreaming || !currentQuestionContext) return;
        const action = btn.dataset.action;
        if (action === 'compare') {
            const message = CHAT_PROMPTS.compare.replace(
                '{choices}', currentQuestionContext.choices.join(', '));
            sendChatMessage(message);
        }
    });
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
    container.scrollTop = container.scrollHeight;
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

    appendChatMessage('user', message);
    const assistantEl = appendChatMessage('assistant', '');
    assistantEl.classList.add('streaming');

    let fullResponse = '';

    try {
        const resp = await fetch(API + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                context: currentQuestionContext,
                history: chatHistory,
            }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });

            for (const line of chunk.split('\n')) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const data = JSON.parse(line.slice(6));
                    if (data.token) {
                        fullResponse += data.token;
                        assistantEl.innerHTML = simpleMarkdown(fullResponse);
                        const container = document.getElementById('chat-messages');
                        container.scrollTop = container.scrollHeight;
                    }
                    if (data.error) {
                        fullResponse += `\n[Error: ${data.error}]`;
                        assistantEl.innerHTML = simpleMarkdown(fullResponse);
                    }
                } catch (e) {
                    // Ignore malformed SSE lines
                }
            }
        }

        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: fullResponse });
        if (fullResponse) addNarrateButton(assistantEl, fullResponse);
    } catch (e) {
        assistantEl.innerHTML = simpleMarkdown(`[Connection error: ${e.message}]`);
    } finally {
        assistantEl.classList.remove('streaming');
        chatStreaming = false;
        document.querySelectorAll('.chat-btn, .chat-send-btn, .word-action-btn').forEach(b => b.disabled = false);
    }
}

// ── Settings ─────────────────────────────────────────────────────────────

async function loadSettings() {
    try {
        const s = await api('/api/settings');
        document.getElementById('set-llm').value = s.llm_provider;
        document.getElementById('set-llm-model').value = s.llm_model;
        document.getElementById('set-tts').value = s.tts_provider;
        document.getElementById('set-tts-voice').value = s.tts_voice;
        document.getElementById('set-session-size').value = s.session_size;
        document.getElementById('set-session-size-val').textContent = s.session_size;
        document.getElementById('set-new-words').value = s.new_words_per_session;
        document.getElementById('set-new-words-val').textContent = s.new_words_per_session;
        document.getElementById('set-min-ready').value = s.min_ready_questions;
        document.getElementById('set-min-ready-val').textContent = s.min_ready_questions;
        document.getElementById('set-max-active').value = s.max_active_words;
        document.getElementById('set-max-active-val').textContent = s.max_active_words;
        document.getElementById('set-archive-interval').value = s.archive_interval_days;
        document.getElementById('set-archive-interval-val').textContent = s.archive_interval_days;
    } catch (e) {
        console.error('Failed to load settings:', e);
    }
}

document.getElementById('set-session-size').addEventListener('input', (e) => {
    document.getElementById('set-session-size-val').textContent = e.target.value;
});

document.getElementById('set-new-words').addEventListener('input', (e) => {
    document.getElementById('set-new-words-val').textContent = e.target.value;
});

document.getElementById('set-min-ready').addEventListener('input', (e) => {
    document.getElementById('set-min-ready-val').textContent = e.target.value;
});

document.getElementById('set-max-active').addEventListener('input', (e) => {
    document.getElementById('set-max-active-val').textContent = e.target.value;
});

document.getElementById('set-archive-interval').addEventListener('input', (e) => {
    document.getElementById('set-archive-interval-val').textContent = e.target.value;
});

document.getElementById('settings-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
        await api('/api/settings', 'PUT', {
            llm_provider: document.getElementById('set-llm').value,
            llm_model: document.getElementById('set-llm-model').value,
            tts_provider: document.getElementById('set-tts').value,
            tts_voice: document.getElementById('set-tts-voice').value,
            session_size: parseInt(document.getElementById('set-session-size').value),
            new_words_per_session: parseInt(document.getElementById('set-new-words').value),
            min_ready_questions: parseInt(document.getElementById('set-min-ready').value),
            max_active_words: parseInt(document.getElementById('set-max-active').value),
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

refreshStats();
