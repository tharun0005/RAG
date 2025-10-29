class ChatbotUI {
    constructor() {
        this.storeId = new URLSearchParams(window.location.search).get('store_id');
        this.usecase = new URLSearchParams(window.location.search).get('usecase');
        this.conversation = [];
        this.selectedFiles = [];
        this.isProcessing = false;
        this.config = this.loadConfig();
        this.abortController = null;
        this.providersData = {};

        console.log('ChatbotUI initialized with storeId:', this.storeId);

        if (!this.validateStoreId()) return;

        this.initialize();
    }

    validateStoreId() {
        if (!this.storeId) {
            this.showNotification('No vector store selected! Redirecting to homepage...', 'error');
            setTimeout(() => window.location.href = '/', 3000);
            return false;
        }
        return true;
    }

    async initialize() {
        try {
            console.log('Initializing chatbot...');
            this.initializeEventListeners();
            await this.loadStoreDetails();
            await this.loadProviders();
            this.updateConfigUI();
            this.updateUI();
            this.showNotification('Chatbot initialized successfully!', 'success');
        } catch (error) {
            console.error('Initialization error:', error);
            this.showNotification('Failed to initialize chatbot', 'error');
        }
    }

    loadConfig() {
        const defaultConfig = {
            llmProvider: '',
            modelName: '',
            topK: 5,
            scoreThreshold: 0.7,
            temperature: 0.7,
            maxTokens: 1000,
            includeSources: true
        };

        try {
            const saved = localStorage.getItem('chatbot-config');
            return saved ? { ...defaultConfig, ...JSON.parse(saved) } : defaultConfig;
        } catch (error) {
            console.error('Error loading config:', error);
            return defaultConfig;
        }
    }

    detectGuardrailViolation(errorMessage, responseData = {}) {
        const errorLower = errorMessage.toLowerCase();

        const violations = {
            sensitive_topic: {
                keywords: ['sensitive', 'topic', 'politics', 'violence', 'sensitive-topic', 'sensitive topic', 'sensitive-topics-guard'],
                type: 'input',
                message: 'Your message contains sensitive content (politics, violence, etc.) that violates our content policy.',
                title: 'Input Blocked - Sensitive Content'
            },
            toxic_language: {
                keywords: ['toxic', 'offensive', 'insulting', 'toxic-language', 'toxic language', 'toxic-language-guard'],
                type: 'output',
                message: 'The AI response was blocked due to potentially toxic or offensive language.',
                title: 'Output Blocked - Toxic Content'
            },
            general_guardrail: {
                keywords: ['guardrail', 'blocked', 'violated', 'policy', 'validation', 'exception', 'failed validation', 'content policy'],
                type: 'general',
                message: 'This request was blocked by our content safety guardrails.',
                title: 'Request Blocked'
            }
        };

        if (responseData.guardrails_blocked === true || responseData.guardrails_passed === false) {
            const guardrailsUsed = responseData.guardrails_used || [];
            if (guardrailsUsed.includes('sensitive-topics-guard') || guardrailsUsed.some(g => g.includes('sensitive'))) {
                return {
                    isGuardrailViolation: true,
                    violationType: 'sensitive_topic',
                    ...violations.sensitive_topic
                };
            } else if (guardrailsUsed.includes('toxic-language-guard') || guardrailsUsed.some(g => g.includes('toxic'))) {
                return {
                    isGuardrailViolation: true,
                    violationType: 'toxic_language',
                    ...violations.toxic_language
                };
            } else {
                return {
                    isGuardrailViolation: true,
                    violationType: 'general_guardrail',
                    ...violations.general_guardrail
                };
            }
        }

        for (const [key, violation] of Object.entries(violations)) {
            if (violation.keywords.some(keyword => errorLower.includes(keyword))) {
                return {
                    isGuardrailViolation: true,
                    violationType: key,
                    ...violation
                };
            }
        }

        return {
            isGuardrailViolation: false,
            type: 'error',
            message: errorMessage,
            title: 'Error'
        };
    }

    async loadStoreDetails() {
        try {
            console.log('Loading store details for:', this.storeId);

            const verifyResponse = await fetch(`/api/vectorstores/${this.storeId}/verify`, {
                signal: this.createAbortSignal()
            });
            const verifyData = await verifyResponse.json();

            if (!verifyResponse.ok || !verifyData.exists) {
                throw new Error(verifyData.error || 'Store not found');
            }

            console.log('Store verified:', verifyData);

            const infoResponse = await fetch(`/api/vectorstores/${this.storeId}/info`, {
                signal: this.createAbortSignal()
            });

            if (!infoResponse.ok) {
                throw new Error('Failed to load store info');
            }

            const storeInfo = await infoResponse.json();
            console.log('Store info loaded:', storeInfo);
            this.updateStoreUI(storeInfo);

        } catch (error) {
            if (error.name === 'AbortError') return;

            console.error('Store load error:', error);
            this.showNotification('Failed to load store details: ' + error.message, 'error');

            if (error.message.includes('not found')) {
                setTimeout(() => window.location.href = '/', 3000);
            }
        }
    }

    updateStoreUI(storeInfo) {
        console.log('Updating UI with store info:', storeInfo);

        const elements = {
            'store-name': storeInfo.name || 'Unknown Store',
            'document-count': storeInfo.documentCount || 0,
            'store-provider': storeInfo.provider || 'Unknown',
            'store-model': this.formatModelName(storeInfo.model) || 'Unknown',
            'created-date': this.formatDate(storeInfo.createdAt) || 'Unknown'
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
                console.log(`Updated ${id}: ${value}`);
            }
        });

        const subtitle = document.getElementById('chat-subtitle');
        if (subtitle) {
            subtitle.textContent = `Using: ${storeInfo.name}`;
        }

        this.updateFileList(storeInfo.files || []);
    }

    updateFileList(files) {
        const fileList = document.getElementById('file-list');
        if (!fileList) return;

        if (files.length === 0) {
            fileList.innerHTML = '<div class="text-gray-400 text-sm">No files uploaded</div>';
            return;
        }

        fileList.innerHTML = files.map(file => {
            const fileName = this.escapeHtml(file.split('/').pop() || file);
            return `
                <div class="file-item" role="listitem">
                    <div class="file-item-info">
                        <i class="fas fa-file file-item-icon" aria-hidden="true"></i>
                        <span class="file-item-name" title="${fileName}">${fileName}</span>
                    </div>
                    <i class="fas fa-check text-green-400 text-xs" aria-label="File uploaded"></i>
                </div>
            `;
        }).join('');
    }

    async loadProviders() {
        try {
            console.log('Loading providers...');
            const response = await fetch('/api/litellm/models', {
                signal: this.createAbortSignal()
            });

            if (!response.ok) throw new Error('Failed to load providers');

            const data = await response.json();
            console.log('Providers loaded:', data);
            this.providersData = this.groupProviders(data.models || []);
            this.populateProviderDropdown();
        } catch (error) {
            if (error.name === 'AbortError') return;

            console.error('Error loading providers:', error);
            this.showNotification('Failed to load AI providers', 'warning');
        }
    }

    groupProviders(models) {
        const providerMap = {};
        models.forEach(model => {
            const provider = model.litellm_provider || 'unknown';
            if (!providerMap[provider]) {
                providerMap[provider] = [];
            }
            providerMap[provider].push(model);
        });
        return providerMap;
    }

    populateProviderDropdown() {
        const providerSelect = document.getElementById('llm-provider');
        const modelSelect = document.getElementById('model-name');

        if (!providerSelect) {
            console.error('Provider select element not found');
            return;
        }

        if (Object.keys(this.providersData).length === 0) {
            providerSelect.innerHTML = '<option value="">No providers available</option>';
            providerSelect.disabled = true;
            return;
        }

        providerSelect.innerHTML = '<option value="">Select a provider</option>';
        Object.keys(this.providersData).forEach(provider => {
            const option = document.createElement('option');
            option.value = provider;
            option.textContent = this.formatProviderName(provider);
            providerSelect.appendChild(option);
        });

        providerSelect.disabled = false;

        providerSelect.addEventListener('change', (e) => {
            this.handleProviderChange(e.target.value);
        });

        if (this.config.llmProvider && this.providersData[this.config.llmProvider]) {
            providerSelect.value = this.config.llmProvider;
            this.handleProviderChange(this.config.llmProvider);
        }
    }

    handleProviderChange(provider) {
        const modelSelect = document.getElementById('model-name');
        if (!modelSelect) return;

        const models = this.providersData[provider] || [];

        modelSelect.innerHTML = '';
        if (models.length > 0) {
            modelSelect.disabled = false;
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_name;
                option.textContent = this.formatModelName(model.model_name);
                option.title = model.model_name;
                modelSelect.appendChild(option);
            });

            const initialModel = this.config.modelName || models[0].model_name;
            if (models.some(m => m.model_name === initialModel)) {
                modelSelect.value = initialModel;
            } else {
                modelSelect.value = models[0].model_name;
                this.updateConfig('modelName', models[0].model_name);
            }
        } else {
            modelSelect.disabled = true;
            modelSelect.innerHTML = '<option value="">No models available</option>';
            this.updateConfig('modelName', '');
        }

        this.updateConfig('llmProvider', provider);
    }

    initializeEventListeners() {
        console.log('Initializing event listeners...');

        const chatForm = document.getElementById('chat-form');
        if (chatForm) {
            chatForm.addEventListener('submit', (e) => this.handleSendMessage(e));
        }

        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    chatForm.dispatchEvent(new Event('submit'));
                }
            });
        }

        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        if (uploadArea && fileInput) {
            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    fileInput.click();
                }
            });

            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
            this.setupDragAndDrop(uploadArea);
        }

        const startUpload = document.getElementById('start-upload');
        if (startUpload) {
            startUpload.addEventListener('click', () => this.uploadFiles());
        }

        this.setupConfigListeners();

        const clearChat = document.getElementById('clear-chat');
        if (clearChat) {
            clearChat.addEventListener('click', () => this.clearChat());
        }

        const downloadChat = document.getElementById('download-chat');
        if (downloadChat) {
            downloadChat.addEventListener('click', () => this.downloadConversation());
        }

        const resetConfig = document.getElementById('reset-config');
        if (resetConfig) {
            resetConfig.addEventListener('click', () => this.resetConfig());
        }

        console.log('Event listeners initialized');
    }

    setupDragAndDrop(uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('upload-dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('upload-dragover');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            this.handleDrop(e);
        });
    }

    setupConfigListeners() {
        const topK = document.getElementById('top-k');
        const scoreThreshold = document.getElementById('score-threshold');
        const temperature = document.getElementById('temperature');
        const maxTokens = document.getElementById('max-tokens');
        const includeSources = document.getElementById('include-sources');
        const modelSelect = document.getElementById('model-name');

        if (topK) {
            topK.addEventListener('change', (e) => {
                this.updateConfig('topK', parseInt(e.target.value));
            });
        }

        if (scoreThreshold) {
            scoreThreshold.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.updateConfig('scoreThreshold', value);
                const display = document.getElementById('threshold-value');
                if (display) {
                    display.textContent = value.toFixed(1);
                    scoreThreshold.setAttribute('aria-valuetext', value.toFixed(1));
                }
            });
        }

        if (temperature) {
            temperature.addEventListener('input', (e) => {
                const value = parseFloat(e.target.value);
                this.updateConfig('temperature', value);
                const display = document.getElementById('temperature-value');
                if (display) {
                    display.textContent = value.toFixed(1);
                    temperature.setAttribute('aria-valuetext', value.toFixed(1));
                }
            });
        }

        if (maxTokens) {
            maxTokens.addEventListener('change', (e) => {
                this.updateConfig('maxTokens', parseInt(e.target.value));
            });
        }

        if (includeSources) {
            includeSources.addEventListener('change', (e) => {
                const isChecked = e.target.checked;
                this.updateConfig('includeSources', isChecked);
                this.showNotification(
                    `Source documents will ${isChecked ? 'be included' : 'not be included'}`,
                    'info'
                );
            });
        }

        if (modelSelect) {
            modelSelect.addEventListener('change', (e) => {
                this.updateConfig('modelName', e.target.value);
            });
        }
    }

    updateConfig(key, value) {
        this.config[key] = value;
        try {
            localStorage.setItem('chatbot-config', JSON.stringify(this.config));
        } catch (error) {
            console.error('Error saving config to localStorage:', error);
        }
    }

    updateConfigUI() {
        const elements = {
            'top-k': this.config.topK,
            'score-threshold': this.config.scoreThreshold,
            'temperature': this.config.temperature,
            'max-tokens': this.config.maxTokens,
            'include-sources': this.config.includeSources
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                if (element.type === 'checkbox') {
                    element.checked = value;
                } else {
                    element.value = value;
                }
            }
        });

        const thresholdValue = document.getElementById('threshold-value');
        if (thresholdValue) {
            thresholdValue.textContent = this.config.scoreThreshold.toFixed(1);
        }

        const tempValue = document.getElementById('temperature-value');
        if (tempValue) {
            tempValue.textContent = this.config.temperature.toFixed(1);
        }
    }

    resetConfig() {
        if (confirm('Are you sure you want to reset all configuration to defaults?')) {
            localStorage.removeItem('chatbot-config');
            this.config = this.loadConfig();
            this.updateConfigUI();

            const providerSelect = document.getElementById('llm-provider');
            const modelSelect = document.getElementById('model-name');

            if (providerSelect) providerSelect.value = '';
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Select a provider first</option>';
                modelSelect.disabled = true;
            }

            this.showNotification('Configuration reset to defaults', 'info');
        }
    }

    handleFileSelect(event) {
        const files = Array.from(event.target.files);
        this.validateAndSetFiles(files);
    }

    handleDrop(event) {
        const files = Array.from(event.dataTransfer.files);
        this.validateAndSetFiles(files);
    }

    validateAndSetFiles(files) {
        const maxSize = 50 * 1024 * 1024;
        const validTypes = ['.pdf', '.txt', '.docx', '.pptx', '.md', '.csv'];

        const validFiles = files.filter(file => {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!validTypes.includes(ext)) {
                this.showNotification(`File type "${ext}" not supported`, 'error');
                return false;
            }
            if (file.size > maxSize) {
                this.showNotification(`File "${file.name}" exceeds 50MB limit`, 'error');
                return false;
            }
            return true;
        });

        this.selectedFiles = validFiles;
        this.updateFilePreview();
    }

    updateFilePreview() {
        const filePreview = document.getElementById('file-preview');
        const selectedFiles = document.getElementById('selected-files');
        const startUpload = document.getElementById('start-upload');

        if (!filePreview || !selectedFiles || !startUpload) return;

        if (this.selectedFiles.length === 0) {
            selectedFiles.classList.add('hidden');
            startUpload.disabled = true;
            return;
        }

        filePreview.innerHTML = this.selectedFiles.map((file, index) => {
            const fileName = this.escapeHtml(file.name);
            return `
                <div class="file-preview-item">
                    <span class="file-name" title="${fileName}">${fileName}</span>
                    <span class="file-size">${this.formatFileSize(file.size)}</span>
                </div>
            `;
        }).join('');

        selectedFiles.classList.remove('hidden');
        startUpload.disabled = false;
    }

    async uploadFiles() {
        if (this.selectedFiles.length === 0) {
            this.showNotification('Please select files to upload', 'error');
            return;
        }

        const startUpload = document.getElementById('start-upload');
        const uploadProgress = document.getElementById('upload-progress');
        const uploadPercentage = document.getElementById('upload-percentage');
        const uploadProgressBar = document.getElementById('upload-progress-bar');

        try {
            startUpload.disabled = true;
            startUpload.setAttribute('aria-busy', 'true');
            uploadProgress.classList.remove('hidden');

            const formData = new FormData();
            this.selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch(`/api/vectorstores/${this.storeId}/files`, {
                method: 'POST',
                body: formData,
                signal: this.createAbortSignal()
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(errorText || 'Upload failed');
            }

            const result = await response.json();
            this.showNotification(`Successfully uploaded ${result.files_uploaded || this.selectedFiles.length} files!`, 'success');

            await this.loadStoreDetails();

            this.selectedFiles = [];
            this.updateFilePreview();

            const fileInput = document.getElementById('file-input');
            if (fileInput) fileInput.value = '';

        } catch (error) {
            if (error.name === 'AbortError') return;

            console.error('Upload error:', error);
            this.showNotification('Failed to upload files: ' + error.message, 'error');
        } finally {
            startUpload.disabled = false;
            startUpload.removeAttribute('aria-busy');
            uploadProgress.classList.add('hidden');

            if (uploadProgressBar) uploadProgressBar.style.width = '0%';
            if (uploadPercentage) uploadPercentage.textContent = '0%';
        }
    }

    async handleSendMessage(event) {
        event.preventDefault();

        const messageInput = document.getElementById('message-input');
        const message = messageInput.value.trim();

        if (!message || this.isProcessing) return;

        const currentConfig = this.getCurrentConfigFromUI();

        if (!currentConfig.modelName) {
            this.showNotification('Please select a provider and model first', 'error');
            return;
        }

        this.isProcessing = true;
        const sendButton = document.getElementById('send-button');

        try {
            this.addMessage('user', message);
            messageInput.value = '';

            if (sendButton) {
                sendButton.disabled = true;
                sendButton.setAttribute('aria-busy', 'true');
            }

            this.showTypingIndicator();

            const requestData = {
                message: message,
                store_id: this.storeId,
                config: {
                    model: currentConfig.modelName,
                    top_k: currentConfig.topK,
                    score_threshold: currentConfig.scoreThreshold,
                    temperature: currentConfig.temperature,
                    max_tokens: currentConfig.maxTokens,
                    include_sources: Boolean(currentConfig.includeSources)
                }
            };

            console.log('Sending chat request:', requestData);

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
                signal: this.createAbortSignal()
            });

            const data = await response.json();

            this.removeTypingIndicator();

            const isGuardrailBlock =
                data.guardrails_blocked === true ||
                data.guardrails_passed === false ||
                (data.response && (
                    data.response.includes("violates our content policy") ||
                    data.response.includes("cannot process this request")
                )) ||
                (data.error && (
                    data.error.toLowerCase().includes('guardrail') ||
                    data.error.toLowerCase().includes('validation') ||
                    data.error.toLowerCase().includes('toxic') ||
                    data.error.toLowerCase().includes('sensitive')
                ));

            if (!response.ok || isGuardrailBlock) {
                const errorMsg = data.error || data.response || 'Request blocked';
                const violation = this.detectGuardrailViolation(errorMsg, data);

                if (violation.isGuardrailViolation || isGuardrailBlock) {
                    this.addGuardrailViolationMessage(violation);
                } else {
                    this.showNotification('Error: ' + errorMsg, 'error');
                }

                this.conversation.push({
                    role: 'user',
                    content: message,
                    timestamp: new Date().toISOString()
                });

                return;
            }

            const chunks = currentConfig.includeSources ? (data.chunks || []) : [];
            console.log('DEBUG: Chunks being passed to addMessage:', chunks);

            this.addMessage('assistant', data.response, chunks, data);

            this.conversation.push(
                { role: 'user', content: message, timestamp: new Date().toISOString() },
                { role: 'assistant', content: data.response, timestamp: new Date().toISOString() }
            );

        } catch (error) {
            if (error.name === 'AbortError') return;

            console.error('Chat error:', error);
            this.removeTypingIndicator();

            const violation = this.detectGuardrailViolation(error.message);
            if (violation.isGuardrailViolation) {
                this.addGuardrailViolationMessage(violation);
            } else {
                this.showNotification('Failed to send message: ' + error.message, 'error');
            }
        } finally {
            this.isProcessing = false;

            if (sendButton) {
                sendButton.disabled = false;
                sendButton.removeAttribute('aria-busy');
            }

            if (messageInput) messageInput.focus();
        }
    }

    getCurrentConfigFromUI() {
        return {
            llmProvider: document.getElementById('llm-provider')?.value || '',
            modelName: document.getElementById('model-name')?.value || '',
            topK: parseInt(document.getElementById('top-k')?.value || '5'),
            scoreThreshold: parseFloat(document.getElementById('score-threshold')?.value || '0.7'),
            temperature: parseFloat(document.getElementById('temperature')?.value || '0.7'),
            maxTokens: parseInt(document.getElementById('max-tokens')?.value || '1000'),
            includeSources: document.getElementById('include-sources')?.checked ?? true
        };
    }

    addGuardrailViolationMessage(violation) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message flex justify-start`;
        messageDiv.setAttribute('role', 'alert');
        messageDiv.setAttribute('aria-live', 'assertive');

        const messageBubble = document.createElement('div');
        messageBubble.className = `max-w-3xl rounded-2xl guardrail-violation ${violation.violationType}`;

        messageBubble.innerHTML = `
            <div class="guardrail-alert">
                <div class="guardrail-header">
                    <div class="guardrail-icon-wrapper">
                        <i class="fas fa-shield-alt guardrail-icon" aria-hidden="true"></i>
                    </div>
                    <div class="guardrail-title">${violation.title}</div>
                </div>
                <div class="guardrail-content">
                    <div class="guardrail-message">${violation.message}</div>
                    <div class="guardrail-type-badge">
                        ${violation.type === 'input' ? 'Input Validation Failed' :
                          violation.type === 'output' ? 'Output Validation Failed' :
                          'Request Blocked by Safety Guardrails'}
                    </div>
                </div>
            </div>
        `;

        messageDiv.appendChild(messageBubble);
        chatMessages.appendChild(messageDiv);

        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    addMessage(role, content, chunks = [], metadata = {}) {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        console.log('DEBUG: addMessage called with role:', role, 'chunks:', chunks);

        const emptyState = document.getElementById('empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;
        messageDiv.setAttribute('role', 'article');
        messageDiv.setAttribute('aria-label', `${role === 'user' ? 'User' : 'Assistant'} message`);

        const messageBubble = document.createElement('div');
        messageBubble.className = `max-w-3xl rounded-2xl p-4 ${
            role === 'user' ? 'message-user' : 'message-assistant'
        }`;

        const timestamp = new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });

        const guardrailsBadge = metadata.guardrails_used && metadata.guardrails_used.length > 0
            ? '<span class="guardrails-badge">Protected</span>'
            : '';

        messageBubble.innerHTML = `
            <div class="message-content">${this.formatMessage(content)}</div>
            <div class="message-footer">
                <span class="message-timestamp">${timestamp}</span>
                ${guardrailsBadge}
            </div>
            ${chunks && chunks.length > 0 ? this.formatSourceDocuments(chunks) : ''}
        `;

        messageDiv.appendChild(messageBubble);
        chatMessages.appendChild(messageDiv);

        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    formatMessage(content) {
        return this.escapeHtml(content)
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }

    formatSourceDocuments(chunks) {
        console.log('=== DEBUG: formatSourceDocuments called ===');
        console.log('DEBUG: chunks parameter type:', typeof chunks);
        console.log('DEBUG: chunks is array?', Array.isArray(chunks));
        console.log('DEBUG: chunks length:', chunks ? chunks.length : 'null/undefined');

        if (!chunks || chunks.length === 0) {
            console.log('DEBUG: No chunks provided or empty array');
            return '';
        }

        console.log('DEBUG: Formatting', chunks.length, 'chunks');
        console.log('DEBUG: Full chunks data:', JSON.stringify(chunks, null, 2));

        return `
            <div class="source-documents">
                <div class="source-documents-title">
                    <i class="fas fa-book" aria-hidden="true"></i>
                    <span>Source Documents (${chunks.length})</span>
                </div>
                ${chunks.map((chunk, index) => {
                    console.log(`DEBUG: Processing chunk ${index + 1}/${chunks.length}`);
                    console.log(`DEBUG: Chunk ${index + 1} raw data:`, chunk);

                    const filename = chunk.file_name ||
                                   chunk.source ||
                                   chunk.metadata?.file_name ||
                                   chunk.metadata?.source ||
                                   'Unknown';

                    const content = chunk.text ||
                                  chunk.content ||
                                  chunk.page_content ||
                                  'No content available';

                    const score = chunk.score !== undefined && chunk.score !== null
                                ? chunk.score.toFixed(3)
                                : 'N/A';

                    const page = chunk.page ||
                               chunk.metadata?.page ||
                               '';

                    console.log(`DEBUG: Chunk ${index + 1} extracted values:`, {
                        filename, content_length: content.length, score, page
                    });

                    return `
                        <div class="source-document">
                            <div class="source-header">
                                <span class="filename" title="${this.escapeHtml(filename)}">
                                    <i class="fas fa-file-alt" aria-hidden="true"></i> ${this.escapeHtml(filename)}
                                </span>
                                <span class="score">Score: ${score}</span>
                            </div>
                            ${page ? `<div class="page-number">Page: ${page}</div>` : ''}
                            <div class="content">${this.escapeHtml(content).substring(0, 200)}${content.length > 200 ? '...' : ''}</div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'chat-message flex justify-start';
        typingDiv.id = 'typing-indicator';
        typingDiv.setAttribute('role', 'status');
        typingDiv.setAttribute('aria-label', 'Assistant is typing');

        typingDiv.innerHTML = `
            <div class="message-assistant max-w-3xl rounded-2xl p-4">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    clearChat() {
        const chatMessages = document.getElementById('chat-messages');
        if (!chatMessages) return;

        if (confirm('Are you sure you want to clear the chat history?')) {
            chatMessages.innerHTML = `
                <div class="text-center text-gray-500 py-8" id="empty-state">
                    <i class="fas fa-robot text-4xl mb-3" aria-hidden="true"></i>
                    <p>Start a conversation by typing a message below</p>
                </div>
            `;
            this.conversation = [];
            this.showNotification('Chat cleared', 'info');
        }
    }

    downloadConversation() {
        if (this.conversation.length === 0) {
            this.showNotification('No conversation to download', 'warning');
            return;
        }

        const chatContent = this.conversation.map(msg => {
            const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : '';
            return `[${timestamp}] ${msg.role.toUpperCase()}: ${msg.content}`;
        }).join('\n\n');

        const metadata = `Chat Session\nVector Store: ${this.storeId}\nUse Case: ${this.usecase || 'N/A'}\nDate: ${new Date().toLocaleString()}\n\n${'='.repeat(60)}\n\n`;

        const blob = new Blob([metadata + chatContent], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-${this.storeId}-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showNotification('Conversation downloaded', 'success');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
    }

    formatDate(dateString) {
        if (!dateString) return 'Unknown';
        try {
            return new Date(dateString).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch {
            return 'Unknown';
        }
    }

    formatProviderName(provider) {
        return provider.charAt(0).toUpperCase() + provider.slice(1).replace(/_/g, ' ');
    }

    formatModelName(model) {
        return model
            .replace(/^sentence-transformers\//, '')
            .replace(/^BAAI\//, '')
            .split('/').pop()
            .replace(/[-_]/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }

    updateUI() {
        const usecaseElement = document.querySelector('#chat-subtitle');
        if (usecaseElement && this.usecase) {
            usecaseElement.textContent += ` - ${this.usecase}`;
        }
    }

    showNotification(message, type = 'info') {
        const existing = document.querySelectorAll('.notification');
        existing.forEach(notif => notif.remove());

        const icons = {
            success: 'check-circle',
            error: 'exclamation-triangle',
            warning: 'exclamation-circle',
            info: 'info-circle'
        };

        const colors = {
            success: 'bg-green-600',
            error: 'bg-red-600',
            warning: 'bg-yellow-600',
            info: 'bg-blue-600'
        };

        const notification = document.createElement('div');
        notification.className = `notification fixed top-4 right-4 p-4 rounded-lg shadow-lg text-white z-[9999] slide-in-right ${colors[type]}`;
        notification.setAttribute('role', 'alert');
        notification.setAttribute('aria-live', 'polite');

        notification.innerHTML = `
            <div class="flex items-center space-x-2">
                <i class="fas fa-${icons[type]}" aria-hidden="true"></i>
                <span>${this.escapeHtml(message)}</span>
            </div>
        `;

        document.body.appendChild(notification);

        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.add('fade-out');
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    createAbortSignal() {
        if (this.abortController) {
            this.abortController.abort();
        }
        this.abortController = new AbortController();
        return this.abortController.signal;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('DOM loaded, initializing ChatbotUI...');
        new ChatbotUI();
    });
} else {
    console.log('DOM already loaded, initializing ChatbotUI...');
    new ChatbotUI();
}
