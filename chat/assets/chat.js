// When the user presses Enter inside the query textarea, trigger a click on the "ask"
// button. We also have to trigger a "change" event on the textarea just before that,
// because otherwise Shiny will debounce changes to the value in the textarea, and the
// value may not be updated before the "ask" button click event happens.
(() => {
  function onDelegatedEvent(eventName, selector, callback) {
    document.addEventListener(eventName, (e) => {
      if (e.target.matches(selector)) {
        callback(e);
      }
    });
  }

  onDelegatedEvent("keydown", ".shiny-gpt-chat textarea", (e) => {
    const { target } = e;
    if (e.code === "Enter" && !e.shiftKey) {
      e.preventDefault();
      target.dispatchEvent(new Event("change"));
      target.disabled = true;
      target.closest(".shiny-gpt-chat").querySelector("button").click();
    }
  });

  onDelegatedEvent("click", ".shiny-gpt-chat button", (e) => {
    e.target.disabled = true;
  });
})();
