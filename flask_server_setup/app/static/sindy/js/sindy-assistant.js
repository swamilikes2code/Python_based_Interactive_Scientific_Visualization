(function () {
  "use strict";

  const THINKING_DELAY_MS = 720;

  function normalize(value) {
    return String(value || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function answerScore(article, query) {
    const normalizedQuery = normalize(query);
    if (!normalizedQuery) return 0;

    const question = normalize(
      article.querySelector(".kb-question")?.textContent
    );
    const keywords = normalize(article.dataset.keywords);
    const haystack = `${question} ${keywords}`;
    const tokens = normalizedQuery.split(" ").filter((token) => token.length > 1);

    let score = haystack.includes(normalizedQuery) ? 20 : 0;
    tokens.forEach((token) => {
      if (question.includes(token)) score += 4;
      else if (keywords.includes(token)) score += 2;
    });
    return score;
  }

  window.initSindyAssistant = function initSindyAssistant() {
    const root = document.querySelector("[data-sindy-assistant]");
    if (!root || root.dataset.initialized === "true") return;
    root.dataset.initialized = "true";

    const chatLog = root.querySelector("[data-chat-log]");
    const searchInput = root.querySelector("[data-assistant-search]");
    const searchButton = root.querySelector("[data-search-submit]");
    const clearButton = root.querySelector("[data-clear-chat]");
    const emptySearch = root.querySelector("[data-empty-search]");
    const cards = Array.from(root.querySelectorAll("[data-question-id]"));
    const articles = Array.from(root.querySelectorAll(".assistant-kb-entry"));
    const articleMap = new Map(articles.map((article) => [article.dataset.id, article]));
    let thinkingTimer = null;
    let busy = false;

    function setBusy(value) {
      busy = value;
      cards.forEach((card) => { card.disabled = value; });
      searchInput.disabled = value;
      searchButton.disabled = value;
      root.classList.toggle("assistant-busy", value);
    }

    function avatarMarkup(extraClass = "") {
      return `<span class="assistant-avatar ${extraClass}" aria-hidden="true">ẋ</span>`;
    }

    function appendUserMessage(text) {
      const message = document.createElement("div");
      message.className = "assistant-message assistant-message-user";
      const bubble = document.createElement("div");
      bubble.className = "assistant-bubble";
      bubble.textContent = text;
      message.appendChild(bubble);
      chatLog.appendChild(message);
    }

    function appendThinkingMessage() {
      const message = document.createElement("div");
      message.className = "assistant-message assistant-message-bot assistant-thinking";
      message.setAttribute("role", "status");
      message.setAttribute("aria-label", "SINDy Assistant is preparing an answer");
      message.innerHTML = `
        ${avatarMarkup("assistant-avatar-thinking")}
        <div class="assistant-bubble">
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
        </div>`;
      chatLog.appendChild(message);
      return message;
    }

    function appendAssistantMessage(article, thinkingMessage) {
      const message = document.createElement("div");
      message.className = "assistant-message assistant-message-bot";

      const content = article.querySelector(".kb-answer").cloneNode(true);
      content.classList.remove("kb-answer");
      content.classList.add("assistant-answer-content");

      const bubble = document.createElement("div");
      bubble.className = "assistant-bubble";
      bubble.appendChild(content);

      const relatedIds = String(article.dataset.related || "")
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      if (relatedIds.length) {
        const related = document.createElement("div");
        related.className = "assistant-related";
        related.innerHTML = "<span>Related questions</span>";
        relatedIds.forEach((id) => {
          const relatedArticle = articleMap.get(id);
          if (!relatedArticle) return;
          const button = document.createElement("button");
          button.type = "button";
          button.className = "assistant-related-chip";
          button.dataset.relatedQuestion = id;
          button.textContent = relatedArticle.querySelector(".kb-question").textContent;
          related.appendChild(button);
        });
        bubble.appendChild(related);
      }

      const note = document.createElement("div");
      note.className = "assistant-answer-note";
      // note.textContent = "Curated SINDy guidance · no data was sent anywhere";
      bubble.appendChild(note);

      message.innerHTML = avatarMarkup();
      message.appendChild(bubble);
      thinkingMessage.replaceWith(message);

      if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise([message]).catch(() => {});
      }
    }

    function scrollToLatest() {
      window.requestAnimationFrame(() => {
        chatLog.scrollTo({ top: chatLog.scrollHeight, behavior: "smooth" });
      });
    }

    function askById(id) {
      if (busy) return;
      const article = articleMap.get(id);
      if (!article) return;

      const question = article.querySelector(".kb-question").textContent.trim();
      appendUserMessage(question);
      const thinkingMessage = appendThinkingMessage();
      clearButton.hidden = false;
      searchInput.value = "";
      filterCards("");
      setBusy(true);
      scrollToLatest();

      thinkingTimer = window.setTimeout(() => {
        if (!document.body.contains(chatLog)) return;
        appendAssistantMessage(article, thinkingMessage);
        setBusy(false);
        scrollToLatest();
      }, THINKING_DELAY_MS);
    }

    function bestMatch(query) {
      return articles
        .map((article) => ({ article, score: answerScore(article, query) }))
        .sort((a, b) => b.score - a.score)[0];
    }

    function submitSearch() {
      const query = searchInput.value.trim();
      if (!query || busy) return;
      const match = bestMatch(query);
      if (match && match.score > 0) {
        askById(match.article.dataset.id);
      } else {
        emptySearch.hidden = false;
        emptySearch.textContent = "No curated answer matched that search. Try a shorter phrase or choose a question below.";
      }
    }

    function filterCards(query) {
      const normalizedQuery = normalize(query);
      let visibleCount = 0;
      cards.forEach((card) => {
        const article = articleMap.get(card.dataset.questionId);
        const haystack = normalize(
          `${card.textContent} ${article ? article.dataset.keywords : ""}`
        );
        const visible = !normalizedQuery || normalizedQuery
          .split(" ")
          .every((token) => haystack.includes(token));
        card.hidden = !visible;
        if (visible) visibleCount += 1;
      });
      root.querySelectorAll("[data-question-group]").forEach((group) => {
        group.hidden = !Array.from(group.querySelectorAll("[data-question-id]"))
          .some((card) => !card.hidden);
      });
      emptySearch.hidden = visibleCount > 0;
      if (!visibleCount) {
        emptySearch.textContent = "No exact topic found. Press Enter and the guide will try the closest curated answer.";
      }
    }

    function resetConversation() {
      if (thinkingTimer) window.clearTimeout(thinkingTimer);
      setBusy(false);
      chatLog.innerHTML = `
        <div class="assistant-message assistant-message-bot assistant-welcome">
          ${avatarMarkup()}
          <div class="assistant-bubble">
            <strong>Hi — I’m the SINDy Research Guide.</strong>
            <p>Choose a question or search the curated knowledge base. I can explain this module’s workflow, metrics, predictions, and ensemble results.</p>
          </div>
        </div>`;
      clearButton.hidden = true;
      searchInput.value = "";
      filterCards("");
    }

    cards.forEach((card) => {
      card.addEventListener("click", () => askById(card.dataset.questionId));
    });
    chatLog.addEventListener("click", (event) => {
      const related = event.target.closest("[data-related-question]");
      if (related) askById(related.dataset.relatedQuestion);
    });
    searchInput.addEventListener("input", () => filterCards(searchInput.value));
    searchInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submitSearch();
      }
    });
    searchButton.addEventListener("click", submitSearch);
    clearButton.addEventListener("click", resetConversation);
  };
})();
