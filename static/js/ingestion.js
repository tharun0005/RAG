class VectorStoreManager {
  constructor() {
    this.currentEditingStore = null;
    this.providersData = {};
    this.lastFocusedElement = null;
    this.abortController = null;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.init();
  }

  init() {
    if (!this.cacheElements()) {
      console.error('Required DOM elements not found');
      return;
    }

    this.bindEvents();
    this.initializeApp();
  }

  cacheElements() {
    const ids = [
      'createVectorBtn', 'createVectorModal', 'closeModal', 'cancelCreate',
      'providerSelect', 'modelSelect', 'vectorStoreForm', 'vectorStoreList',
      'noStores', 'vectorName', 'description', 'fileUpload', 'submitCreate',
      'dropZone', 'fileList', 'fileListItems', 'uploadProgress', 'progressBar',
      'progressPercent', 'uploadStatus', 'main-content'
    ];

    this.elements = {};
    for (const id of ids) {
      this.elements[id] = document.getElementById(id);
      if (!this.elements[id] && !['dropZone', 'fileList', 'uploadProgress'].includes(id)) {
        console.error(`Element not found: ${id}`);
        return false;
      }
    }
    return true;
  }

  bindEvents() {
    this.elements.createVectorBtn.addEventListener('click', () => this.showModal());
    this.elements.closeModal.addEventListener('click', () => this.hideModal());
    this.elements.cancelCreate.addEventListener('click', () => this.hideModal());

    this.elements.createVectorModal.addEventListener('click', (e) => {
      if (e.target === this.elements.createVectorModal) {
        this.hideModal();
      }
    });

    this.elements.providerSelect.addEventListener('change', (e) => {
      this.loadModels(e.target.value);
    });

    this.elements.vectorStoreForm.addEventListener('submit', (e) => {
      this.handleFormSubmit(e);
    });

    this.elements.fileUpload.addEventListener('change', (e) => {
      this.handleFileSelect(e.target.files);
    });

    if (this.elements.dropZone) {
      this.setupDragAndDrop();
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.elements.createVectorModal.classList.contains('hidden')) {
        this.hideModal();
      }
    });

    this.elements.createVectorModal.addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        this.trapFocus(e);
      }
    });
  }

  setupDragAndDrop() {
    const dropZone = this.elements.dropZone;
    const fileInput = this.elements.fileUpload;

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
      });
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.add('drag-over');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropZone.addEventListener(eventName, () => {
        dropZone.classList.remove('drag-over');
      });
    });

    dropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      fileInput.files = files;
      this.handleFileSelect(files);
    });

    dropZone.addEventListener('click', () => {
      fileInput.click();
    });

    dropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        fileInput.click();
      }
    });
  }

  handleFileSelect(files) {
    this.clearError('fileUpload');

    const validation = this.validateFiles(files);
    if (!validation.valid) {
      this.showError('fileUpload', validation.message);
      return;
    }

    this.updateFileList(files);
  }

  validateFiles(files) {
    const maxFiles = 10;
    const maxSize = 50 * 1024 * 1024;
    const allowedTypes = ['.pdf', '.txt', '.docx', '.doc'];

    if (files.length === 0) {
      return { valid: false, message: 'Please select at least one file' };
    }

    if (files.length > maxFiles) {
      return { valid: false, message: `Maximum ${maxFiles} files allowed` };
    }

    for (const file of files) {
      if (file.size > maxSize) {
        return { valid: false, message: `File "${file.name}" exceeds 50MB limit` };
      }

      const ext = '.' + file.name.split('.').pop().toLowerCase();
      if (!allowedTypes.includes(ext)) {
        return { valid: false, message: `File type "${ext}" not supported` };
      }
    }

    return { valid: true };
  }

  updateFileList(files) {
    if (!this.elements.fileList) return;

    this.elements.fileListItems.innerHTML = '';

    if (files.length === 0) {
      this.elements.fileList.classList.add('hidden');
      return;
    }

    this.elements.fileList.classList.remove('hidden');

    Array.from(files).forEach((file, index) => {
      const li = document.createElement('li');
      li.className = 'file-item';
      li.innerHTML = `
        <div class="file-item-info">
          <span class="file-item-name" title="${this.escapeHtml(file.name)}">${this.escapeHtml(file.name)}</span>
          <span class="file-item-size">${this.formatFileSize(file.size)}</span>
        </div>
        <button
          type="button"
          class="file-item-remove"
          data-index="${index}"
          aria-label="Remove ${this.escapeHtml(file.name)}">
          ×
        </button>
      `;

      const removeBtn = li.querySelector('.file-item-remove');
      removeBtn.addEventListener('click', () => {
        this.removeFile(index);
      });

      this.elements.fileListItems.appendChild(li);
    });
  }

  removeFile(index) {
    const dt = new DataTransfer();
    const files = Array.from(this.elements.fileUpload.files);

    files.forEach((file, i) => {
      if (i !== index) {
        dt.items.add(file);
      }
    });

    this.elements.fileUpload.files = dt.files;
    this.updateFileList(dt.files);
  }

  formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  }

  trapFocus(e) {
    const modal = this.elements.createVectorModal;
    const focusable = modal.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  showModal() {
    this.lastFocusedElement = document.activeElement;

    this.resetForm();
    this.elements.createVectorModal.querySelector('h2').textContent = 'Create Vector Store';
    this.elements.submitCreate.textContent = 'Create Store';
    this.currentEditingStore = null;

    this.elements.createVectorModal.classList.remove('hidden');
    this.elements['main-content']?.setAttribute('aria-hidden', 'true');
    document.body.classList.add('modal-open');

    setTimeout(() => {
      this.elements.vectorName.focus();
    }, 100);
  }

  hideModal() {
    this.elements.createVectorModal.classList.add('hidden');
    this.elements['main-content']?.removeAttribute('aria-hidden');
    document.body.classList.remove('modal-open');

    this.resetForm();
    this.currentEditingStore = null;

    if (this.lastFocusedElement) {
      this.lastFocusedElement.focus();
    }
  }

  resetForm() {
    this.elements.vectorStoreForm.reset();
    this.elements.modelSelect.innerHTML = '<option value="">— Select provider first —</option>';
    this.elements.modelSelect.disabled = true;
    this.enableFormFields();
    this.clearAllErrors();

    if (this.elements.fileList) {
      this.elements.fileList.classList.add('hidden');
      this.elements.fileListItems.innerHTML = '';
    }
  }

  enableFormFields() {
    this.elements.vectorName.disabled = false;
    this.elements.providerSelect.disabled = false;
    this.elements.description.disabled = false;
    this.elements.fileUpload.disabled = false;
  }

  async loadProviders() {
    try {
      const endpoints = [
        '/api/embedding-config',
        '/data/embedding_config.json',
        '/static/data/embedding_config.json'
      ];

      let res = null;
      for (const endpoint of endpoints) {
        try {
          res = await fetch(endpoint, { signal: this.createAbortSignal() });
          if (res.ok) break;
        } catch (e) {
          if (e.name === 'AbortError') throw e;
        }
      }

      if (!res || !res.ok) {
        this.providersData = this.getDefaultProviders();
      } else {
        this.providersData = await res.json();
      }

      this.populateProviderSelect();

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Failed to load providers:', err);
        this.providersData = this.getDefaultProviders();
        this.populateProviderSelect();
      }
    }
  }

  getDefaultProviders() {
    return {
      "huggingface": {
        "embedding_models": [
          "BAAI/bge-small-en",
          "sentence-transformers/all-mpnet-base-v2",
          "sentence-transformers/all-MiniLM-L6-v2"
        ]
      },
      "ollama": {
        "embedding_models": [
          "nomic-embed-text",
          "all-minilm"
        ]
      }
    };
  }

  populateProviderSelect() {
    this.elements.providerSelect.innerHTML = '<option value="">— Select provider —</option>';

    if (Object.keys(this.providersData).length === 0) {
      this.elements.providerSelect.innerHTML = '<option value="">— No providers available —</option>';
      this.elements.providerSelect.disabled = true;
      return;
    }

    for (const provider in this.providersData) {
      const opt = document.createElement('option');
      opt.value = provider;
      opt.textContent = this.formatProviderName(provider);
      this.elements.providerSelect.appendChild(opt);
    }

    this.elements.providerSelect.disabled = false;
  }

  formatProviderName(provider) {
    return provider.charAt(0).toUpperCase() + provider.slice(1).replace(/([A-Z])/g, ' $1');
  }

  async loadModels(provider) {
    this.elements.modelSelect.innerHTML = '<option value="">— Select model —</option>';

    if (!provider || !this.providersData[provider]) {
      this.elements.modelSelect.disabled = true;
      return;
    }

    const models = this.providersData[provider].embedding_models;

    if (!models || !models.length) {
      this.elements.modelSelect.innerHTML = '<option value="">— No models available —</option>';
      this.elements.modelSelect.disabled = true;
      return;
    }

    this.elements.modelSelect.disabled = false;

    models.forEach(model => {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = this.formatModelName(model);
      opt.title = model;
      this.elements.modelSelect.appendChild(opt);
    });
  }

  formatModelName(model) {
    return model
      .replace(/^sentence-transformers\//, '')
      .replace(/^BAAI\//, '')
      .split('/').pop()
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase());
  }

  async loadVectorStores() {
    try {
      const res = await fetch('/api/vectorstores', {
        signal: this.createAbortSignal()
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const data = await res.json();

      this.elements.vectorStoreList.innerHTML = '';

      if (!data.vector_stores || !data.vector_stores.length) {
        this.elements.noStores.classList.remove('hidden');
        this.elements.vectorStoreList.appendChild(this.elements.noStores);
        return;
      }

      this.elements.noStores.classList.add('hidden');

      data.vector_stores.forEach(store => {
        const card = this.createVectorStoreCard(store);
        this.elements.vectorStoreList.appendChild(card);
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Failed to load vector stores:', err);
        this.elements.noStores.classList.remove('hidden');
        this.elements.vectorStoreList.appendChild(this.elements.noStores);
        this.showNotification('Error loading vector stores', 'error');
      }
    }
  }

  createVectorStoreCard(store) {
    const card = document.createElement('button');
    card.className = 'vector-store-card';
    card.type = 'button';
    card.setAttribute('data-store-id', store.id || store.name);
    card.setAttribute('aria-label', `Vector store: ${store.name}`);

    const formattedModel = this.formatModelName(store.model);

    const fileCount = store.fileCount || 0;
    const documentCount = store.documentCount || 0;

    const fileLabel = fileCount === 1 ? 'File' : 'Files';
    const docLabel = documentCount === 1 ? 'Chunk' : 'Chunks';

    card.innerHTML = `
      <div class="card-header">
        <h3 class="store-name">${this.escapeHtml(store.name)}</h3>
        <span class="provider-badge">${this.formatProviderName(store.provider)}</span>
      </div>

      <div class="card-content">
        <div class="model-info">
          <span class="model-icon" aria-hidden="true">🤖</span>
          <span title="${this.escapeHtml(store.model)}">${formattedModel}</span>
        </div>

        ${store.description ? `
          <div class="description" title="${this.escapeHtml(store.description)}">
            ${this.escapeHtml(store.description)}
          </div>
        ` : ''}
      </div>

      <div class="card-stats">
        <div class="stat">
          <span class="stat-value">${fileCount}</span>
          <span class="stat-label">${fileLabel}</span>
        </div>
        <div class="stat">
          <span class="stat-value">${documentCount}</span>
          <span class="stat-label">${docLabel}</span>
        </div>
      </div>

      <div class="date-info">
        Created ${this.formatDate(store.createdAt || store.created_at)}
      </div>

      <div class="card-actions">
        <button type="button" class="action-btn upload-btn" data-id="${store.id || store.name}">
          📁 Upload Files
        </button>
        <button type="button" class="action-btn delete-btn" data-id="${store.id || store.name}">
          🗑️ Delete Store
        </button>
      </div>
    `;

    card.addEventListener('click', function(e) {
      if (!e.target.closest('.card-actions')) {
        this.classList.toggle('active');
      }
    });

    const uploadBtn = card.querySelector('.upload-btn');
    const deleteBtn = card.querySelector('.delete-btn');

    uploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.openUploadModal(uploadBtn.dataset.id, store);
    });

    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.deleteVectorStore(deleteBtn.dataset.id, store.name);
    });

    return card;
  }

  openUploadModal(storeId, store) {
    this.currentEditingStore = storeId;

    this.elements.vectorName.value = store.name;
    this.elements.vectorName.disabled = true;

    this.elements.providerSelect.value = store.provider;
    this.elements.providerSelect.disabled = true;

    this.loadModels(store.provider).then(() => {
      this.elements.modelSelect.value = store.model;
      this.elements.modelSelect.disabled = true;
    });

    this.elements.description.value = store.description || '';
    this.elements.description.disabled = true;

    this.elements.createVectorModal.querySelector('h2').textContent = 'Upload Files to Vector Store';
    this.elements.submitCreate.textContent = 'Upload Files';

    this.elements.createVectorModal.classList.remove('hidden');
    this.elements['main-content']?.setAttribute('aria-hidden', 'true');
    document.body.classList.add('modal-open');
    this.lastFocusedElement = document.activeElement;

    setTimeout(() => {
      if (this.elements.dropZone) {
        this.elements.dropZone.focus();
      } else {
        this.elements.fileUpload.focus();
      }
    }, 100);
  }

  async deleteVectorStore(storeId, storeName) {
    if (!confirm(`Are you sure you want to delete "${storeName}"? This cannot be undone.`)) {
      return;
    }

    try {
      const res = await fetch(`/api/vectorstores/${storeId}`, {
        method: 'DELETE',
        signal: this.createAbortSignal()
      });

      if (res.ok) {
        this.showNotification(`Vector store "${storeName}" deleted successfully!`, 'success');
        this.loadVectorStores();
      } else {
        const errorText = await res.text();
        throw new Error(errorText || 'Delete failed');
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Failed to delete vector store:', err);
        this.showNotification('Error deleting vector store', 'error');
      }
    }
  }

  async handleFormSubmit(e) {
    e.preventDefault();

    this.clearAllErrors();

    const vectorName = this.elements.vectorName.value.trim();
    const provider = this.elements.providerSelect.value;
    const model = this.elements.modelSelect.value;
    const description = this.elements.description.value.trim();
    const files = this.elements.fileUpload.files;

    if (!vectorName) {
      this.showError('vectorName', 'Please enter a vector store name');
      this.elements.vectorName.focus();
      return;
    }

    if (!provider) {
      this.showError('providerSelect', 'Please select a provider');
      this.elements.providerSelect.focus();
      return;
    }

    if (!model) {
      this.showError('modelSelect', 'Please select an embedding model');
      this.elements.modelSelect.focus();
      return;
    }

    const fileValidation = this.validateFiles(files);
    if (!fileValidation.valid) {
      this.showError('fileUpload', fileValidation.message);
      return;
    }

    if (!this.currentEditingStore && files.length === 0) {
      this.showError('fileUpload', 'Please select at least one file');
      return;
    }

    try {
      this.setLoadingState(true);

      if (this.currentEditingStore) {
        await this.uploadFilesToStore(this.currentEditingStore, files);
      } else {
        await this.createNewVectorStore({
          name: vectorName,
          provider,
          model,
          description,
          files
        });
      }

      this.hideModal();
      this.loadVectorStores();

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Form submission error:', err);
        this.showNotification('Error processing your request', 'error');
      }
    } finally {
      this.setLoadingState(false);
    }
  }

  setLoadingState(loading) {
    const submitBtn = this.elements.submitCreate;
    const form = this.elements.vectorStoreForm;

    if (loading) {
      submitBtn.disabled = true;
      submitBtn.setAttribute('aria-busy', 'true');
      form.querySelectorAll('input, select, textarea, button').forEach(el => {
        if (el.id !== 'closeModal' && el.id !== 'cancelCreate') {
          el.disabled = true;
        }
      });

      if (this.elements.uploadProgress) {
        this.elements.uploadProgress.classList.remove('hidden');
      }
    } else {
      submitBtn.disabled = false;
      submitBtn.removeAttribute('aria-busy');

      if (!this.currentEditingStore) {
        this.enableFormFields();
        this.elements.modelSelect.disabled = !this.elements.providerSelect.value;
      }

      if (this.elements.uploadProgress) {
        this.elements.uploadProgress.classList.add('hidden');
      }
    }
  }

  async createNewVectorStore(storeData) {
    const formData = new FormData();
    formData.append('name', storeData.name);
    formData.append('provider', storeData.provider);
    formData.append('model', storeData.model);
    formData.append('description', storeData.description);

    for (let i = 0; i < storeData.files.length; i++) {
      formData.append('files', storeData.files[i]);
    }

    const res = await fetch('/api/vectorstores', {
      method: 'POST',
      body: formData,
      signal: this.createAbortSignal()
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(errorText || 'Create failed');
    }

    this.showNotification(`Vector store "${storeData.name}" created successfully!`, 'success');
    return await res.json();
  }

  async uploadFilesToStore(storeId, files) {
    if (files.length === 0) {
      throw new Error('No files selected');
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    const res = await fetch(`/api/vectorstores/${storeId}/files`, {
      method: 'POST',
      body: formData,
      signal: this.createAbortSignal()
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(errorText || 'Upload failed');
    }

    this.showNotification('Files uploaded successfully!', 'success');
  }

  showError(fieldId, message) {
    const field = this.elements[fieldId];
    const errorId = `${fieldId}-error`;
    const errorElement = document.getElementById(errorId);

    if (field) {
      field.setAttribute('aria-invalid', 'true');
      field.classList.add('border-red-500');
    }

    if (errorElement) {
      errorElement.textContent = message;
      errorElement.classList.remove('hidden');
    }
  }

  clearError(fieldId) {
    const field = this.elements[fieldId];
    const errorId = `${fieldId}-error`;
    const errorElement = document.getElementById(errorId);

    if (field) {
      field.removeAttribute('aria-invalid');
      field.classList.remove('border-red-500');
    }

    if (errorElement) {
      errorElement.textContent = '';
      errorElement.classList.add('hidden');
    }
  }

  clearAllErrors() {
    const errorElements = document.querySelectorAll('[id$="-error"]');
    errorElements.forEach(el => {
      el.textContent = '';
      el.classList.add('hidden');
    });

    const fields = document.querySelectorAll('[aria-invalid="true"]');
    fields.forEach(field => {
      field.removeAttribute('aria-invalid');
      field.classList.remove('border-red-500');
    });
  }

  formatDate(dateString) {
    if (!dateString) return 'Unknown date';

    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) return 'Invalid date';

      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    } catch {
      return 'Unknown date';
    }
  }

  showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notification = document.createElement('div');
    notification.className = `notification fixed top-4 right-4 px-6 py-3 rounded-xl z-[9999] shadow-lg ${
      type === 'success' ? 'bg-green-500 text-white' :
      type === 'error' ? 'bg-red-500 text-white' :
      'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    notification.setAttribute('role', 'alert');
    notification.setAttribute('aria-live', 'polite');

    document.body.appendChild(notification);

    notification.offsetHeight;
    notification.classList.add('show');

    setTimeout(() => {
      notification.classList.remove('show');
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  }

  createParticles() {
    if (this.reducedMotion) return;

    const container = document.getElementById('particles');
    if (!container) return;

    const count = 30;

    for (let i = 0; i < count; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';

      const size = Math.random() * 4 + 2;
      const left = Math.random() * 100;
      const duration = Math.random() * 15 + 10;
      const delay = Math.random() * 5;

      Object.assign(particle.style, {
        width: `${size}px`,
        height: `${size}px`,
        left: `${left}%`,
        animationDuration: `${duration}s`,
        animationDelay: `${delay}s`
      });

      container.appendChild(particle);
    }
  }

  createAbortSignal() {
    if (this.abortController) {
      this.abortController.abort();
    }
    this.abortController = new AbortController();
    return this.abortController.signal;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  async initializeApp() {
    try {
      await this.loadProviders();
      await this.loadVectorStores();
      this.createParticles();
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Initialization error:', err);
      }
    }
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new VectorStoreManager());
} else {
  new VectorStoreManager();
}
