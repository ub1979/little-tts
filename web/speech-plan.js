/* Keep sentence context when it fits; split oversized requests at phrase boundaries. */
(function (root) {
  function planSpeech(text, phonemize, tokenLimit) {
    const limit = Math.floor(tokenLimit);
    if (!Number.isFinite(limit) || limit < 8) throw new Error("Invalid phoneme limit.");
    const input = String(text).replace(/\s+/g, " ").trim();
    if (!/[\p{L}\p{N}]/u.test(input)) throw new Error("Enter some words to speak.");
    function fit(part) {
      const phones = phonemize(part);
      if (phones.ids.length <= limit) return [{ ...phones, inputText: part }];
      // Keep punctuation with its phrase, so the model still predicts its pause.
      const pieces = part.split(/(?<=[,;:!?…])\s+|(?<=\.)\s+(?=[A-Z])/u);
      const phrases = [];
      const abbreviation = /(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$/i;
      for (const piece of pieces) {
        const previous = phrases.length - 1;
        if (previous >= 0 && abbreviation.test(phrases[previous])) phrases[previous] += " " + piece;
        else phrases.push(piece);
      }
      if (phrases.length > 1) {
        const chunks = [];
        let pending = "";
        for (const phrase of phrases) {
          const combined = pending ? pending + " " + phrase : phrase;
          if (phonemize(combined).ids.length <= limit) pending = combined;
          else {
            if (pending) chunks.push(...fit(pending));
            pending = "";
            chunks.push(...fit(phrase));
          }
        }
        if (pending) chunks.push(...fit(pending));
        return chunks;
      }
      const words = part.split(" ");
      if (words.length < 2) throw new Error("A word is too long to pronounce. Try a shorter spelling.");
      const middle = Math.ceil(words.length / 2);
      return [...fit(words.slice(0, middle).join(" ")), ...fit(words.slice(middle).join(" "))];
    }
    return fit(input);
  }
  root.planSpeech = planSpeech;
  if (typeof module !== "undefined" && module.exports) module.exports = { planSpeech };
})(typeof self !== "undefined" ? self : globalThis);
