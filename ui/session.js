/* Anonymous access key, persisted per browser and site. Never put it in a URL. */
window.createHydroSession = function(apiBase) {
  const key = "hb-session-v1";
  let pending = null;

  function stored() {
    try {
      return window.localStorage.getItem(key);
    } catch (e) {
      throw new Error("Дозвольте збереження даних цього сайту в браузері, щоб відкрити ваші поля.");
    }
  }

  async function createOrRestore() {
    const existing = stored();
    if (existing) return existing;
    let response;
    try {
      response = await window.fetch(apiBase + "/sessions", {method: "POST", cache: "no-store"});
    } catch (e) {
      throw new Error("Не вдалося відкрити сесію. Перевірте з’єднання та оновіть сторінку.");
    }
    if (!response.ok) throw new Error("Не вдалося відкрити сесію. Спробуйте трохи пізніше.");
    const body = await response.json();
    if (!/^[A-Za-z0-9_-]{43}$/.test(body.token || "")) {
      throw new Error("Сервер не підтвердив сесію. Спробуйте оновити сторінку.");
    }
    // A second tab may already have finished creating this browser's session.
    if (stored()) return stored();
    try {
      window.localStorage.setItem(key, body.token);
      if (stored() !== body.token) throw new Error("storage unavailable");
    } catch (e) {
      throw new Error("Не вдалося зберегти доступ до полів. Дозвольте збереження даних сайту в браузері.");
    }
    return body.token;
  }

  return async function token() {
    const existing = stored();
    if (existing) return existing;
    if (!pending) {
      // Serialize the first visit across tabs as well as within this page.
      pending = (window.navigator.locks
        ? window.navigator.locks.request("hydrobrain-session-init", createOrRestore)
        : createOrRestore()).finally(() => { pending = null; });
    }
    return pending;
  };
};
