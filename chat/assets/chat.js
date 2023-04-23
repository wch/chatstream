// When the user presses Enter inside the query textarea, trigger a click on the "ask"
// button. We also have to trigger a "change" event on the textarea just before that,
// because otherwise Shiny will debounce changes to the value in the textarea, and the
// value may not be updated before the "ask" button click event happens.
(() => {
  document.addEventListener("keydown", (e) => {
    const { target } = e;
    if (target.matches(".shiny-gpt-chat textarea")) {
      if (e.code === "Enter" && !e.shiftKey) {
        e.preventDefault();
        target.dispatchEvent(new Event("change"));
        target.disabled = true;
        target.closest(".shiny-gpt-chat").querySelector("button").click();
      }
    }
  });

  document.addEventListener("input", (e) => {
    const { target } = e;
    if (target.matches(".shiny-gpt-chat textarea")) {
      // Automatically resize the textarea to fit its content.
      target.style.height = "auto";
      target.style.height = target.scrollHeight + "px";
    }
  });
})();
