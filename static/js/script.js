const CONFIG = {
  PARTICLE_INTERVAL: 2000,
  PARTICLE_BATCH_SIZE: 3,
  MAX_PARTICLES: 50,
  API_TIMEOUT: 10000,
  REDUCED_MOTION_QUERY: '(prefers-reduced-motion: reduce)'
};

class DashboardManager {
  constructor() {
    this.elements = {};
    this.particleCount = 0;
    this.reducedMotion = window.matchMedia(CONFIG.REDUCED_MOTION_QUERY).matches;
    this.init();
  }

  init() {
    if (!this.cacheElements()) {
      console.error('Required DOM elements not found');
      return;
    }

    this.bindEvents();
    this.setupAccessibility();

    if (!this.reducedMotion) {
      this.startParticles();
    }
  }

  cacheElements() {
    const ids = [
      'dataCard', 'chatCard', 'chatbot-modal', 'close-modal',
      'cancel-chatbot', 'chatbot-form', 'vector-store-select',
      'usecase-name', 'particles', 'main-content', 'submit-chatbot',
      'form-loading'
    ];

    for (const id of ids) {
      this.elements[id] = document.getElementById(id);
      if (!this.elements[id] && id !== 'particles') {
        console.error(`Element not found: ${id}`);
        return false;
      }
    }
    return true;
  }

  bindEvents() {
    this.elements.dataCard.addEventListener('click', () => {
      this.navigateToDataIngestion();
    });

    this.elements.chatCard.addEventListener('click', () => {
      this.showChatbotModal();
    });

    this.elements['close-modal'].addEventListener('click', () => {
      this.hideModal();
    });

    this.elements['cancel-chatbot'].addEventListener('click', () => {
      this.hideModal();
    });

    this.elements['chatbot-modal'].addEventListener('click', (e) => {
      if (e.target === this.elements['chatbot-modal']) {
        this.hideModal();
      }
    });

    this.elements['chatbot-form'].addEventListener('submit', (e) => {
      this.handleFormSubmit(e);
    });

    if (!this.reducedMotion) {
      this.setupCardHoverEffects();
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && !this.elements['chatbot-modal'].classList.contains('hidden')) {
        this.hideModal();
      }
    });

    window.matchMedia(CONFIG.REDUCED_MOTION_QUERY).addEventListener('change', (e) => {
      this.reducedMotion = e.matches;
    });
  }

  setupAccessibility() {
    this.elements['chatbot-modal'].addEventListener('keydown', (e) => {
      if (e.key === 'Tab') {
        this.trapFocus(e);
      }
    });
  }

  trapFocus(e) {
    const modal = this.elements['chatbot-modal'];
    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    if (e.shiftKey && document.activeElement === firstElement) {
      e.preventDefault();
      lastElement.focus();
    } else if (!e.shiftKey && document.activeElement === lastElement) {
      e.preventDefault();
      firstElement.focus();
    }
  }

  navigateToDataIngestion() {
    window.location.href = '/data-ingestion';
  }

  async showChatbotModal() {
    this.lastFocusedElement = document.activeElement;

    await this.loadVectorStores();

    this.elements['chatbot-modal'].classList.remove('hidden');
    this.elements['main-content'].setAttribute('aria-hidden', 'true');
    document.body.classList.add('modal-open');

    setTimeout(() => {
      this.elements['vector-store-select'].focus();
    }, 100);
  }

  hideModal() {
    this.elements['chatbot-modal'].classList.add('hidden');
    this.elements['main-content'].removeAttribute('aria-hidden');
    document.body.classList.remove('modal-open');

    this.elements['chatbot-form'].reset();
    this.clearErrors();

    if (this.lastFocusedElement) {
      this.lastFocusedElement.focus();
    }
  }

  async loadVectorStores() {
    const select = this.elements['vector-store-select'];

    try {
      select.innerHTML = '<option value="">Loading...</option>';
      select.disabled = true;

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);

      const response = await fetch('/api/vectorstores', {
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const stores = data.vector_stores || [];

      select.innerHTML = stores.length === 0
        ? '<option value="">No vector stores available</option>'
        : '<option value="">Select a vector store...</option>';

      stores.forEach(store => {
        const option = document.createElement('option');
        option.value = store.id;

        const fileCount = store.fileCount || store.files?.length || 0;
        const fileLabel = fileCount === 1 ? 'file' : 'files';

        option.textContent = `${store.name} (${fileCount} ${fileLabel})`;
        option.title = `${store.name} - ${fileCount} files, ${store.documentCount || 0} chunks`;

        select.appendChild(option);
      });

      select.disabled = false;

    } catch (error) {
      console.error('Error loading vector stores:', error);

      select.innerHTML = error.name === 'AbortError'
        ? '<option value="">Request timeout - please try again</option>'
        : '<option value="">Error loading stores</option>';

      this.showError('vector-store-select', 'Failed to load vector stores');
      select.disabled = false;
    }
  }

  async handleFormSubmit(e) {
    e.preventDefault();

    const selectedStore = this.elements['vector-store-select'].value;
    const usecase = this.elements['usecase-name'].value.trim();

    this.clearErrors();

    if (!selectedStore) {
      this.showError('vector-store-select', 'Please select a vector store');
      return;
    }

    if (!usecase) {
      this.showError('usecase-name', 'Please enter a use case name');
      return;
    }

    this.setLoadingState(true);

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);

      const response = await fetch(`/api/vectorstores/${selectedStore}/verify`, {
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Failed to verify vector store');
      }

      const verifyData = await response.json();

      if (!verifyData.exists) {
        this.showError('vector-store-select', 'Vector store does not exist or has no documents');
        return;
      }

      if (!verifyData.has_embeddings) {
        const proceed = confirm('No embeddings found. Continue anyway?');
        if (!proceed) return;
      }

      window.location.href = `/chatbot-ui?store_id=${selectedStore}&usecase=${encodeURIComponent(usecase)}`;

    } catch (error) {
      console.error('Error creating chatbot:', error);

      const message = error.name === 'AbortError'
        ? 'Request timeout - please try again'
        : 'Error creating chatbot - please try again';

      this.showError('usecase-name', message);
    } finally {
      this.setLoadingState(false);
    }
  }

  setLoadingState(loading) {
    const submitBtn = this.elements['submit-chatbot'];
    const loadingDiv = this.elements['form-loading'];
    const form = this.elements['chatbot-form'];

    if (loading) {
      submitBtn.disabled = true;
      submitBtn.setAttribute('aria-busy', 'true');
      loadingDiv?.classList.remove('hidden');
      form.querySelectorAll('input, select').forEach(el => el.disabled = true);
    } else {
      submitBtn.disabled = false;
      submitBtn.removeAttribute('aria-busy');
      loadingDiv?.classList.add('hidden');
      form.querySelectorAll('input, select').forEach(el => el.disabled = false);
    }
  }

  showError(fieldId, message) {
    const field = document.getElementById(fieldId);
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

  clearErrors() {
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

  setupCardHoverEffects() {
    const cards = [this.elements.dataCard, this.elements.chatCard];

    cards.forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.willChange = 'transform';
      });

      card.addEventListener('mousemove', this.throttle((e) => {
        const rect = card.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width - 0.5;
        const y = (e.clientY - rect.top) / rect.height - 0.5;

        requestAnimationFrame(() => {
          card.style.transform = `perspective(1000px) rotateY(${x * 10}deg) rotateX(${-y * 10}deg) scale(1.02)`;
        });
      }, 16));

      card.addEventListener('mouseleave', () => {
        requestAnimationFrame(() => {
          card.style.transform = '';
          card.style.willChange = 'auto';
        });
      });
    });
  }

  startParticles() {
    if (!this.elements.particles) return;

    const createParticle = () => {
      if (this.particleCount >= CONFIG.MAX_PARTICLES) return;

      const particle = document.createElement('div');
      particle.className = 'particle';

      const size = Math.random() * 4 + 3;
      const duration = Math.random() * 3 + 5;
      const hue = 180 + Math.random() * 120;

      Object.assign(particle.style, {
        width: `${size}px`,
        height: `${size}px`,
        left: `${Math.random() * 100}%`,
        animationDuration: `${duration}s`,
        background: `radial-gradient(circle, hsla(${hue}, 100%, 70%, 0.6), transparent)`
      });

      this.elements.particles.appendChild(particle);
      this.particleCount++;

      setTimeout(() => {
        particle.remove();
        this.particleCount--;
      }, duration * 1000);
    };

    setInterval(() => {
      for (let i = 0; i < CONFIG.PARTICLE_BATCH_SIZE; i++) {
        createParticle();
      }
    }, CONFIG.PARTICLE_INTERVAL);
  }

  throttle(func, delay) {
    let lastCall = 0;
    return function(...args) {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        func.apply(this, args);
      }
    };
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new DashboardManager());
} else {
  new DashboardManager();
}
