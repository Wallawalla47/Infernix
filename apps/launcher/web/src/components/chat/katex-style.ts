// KaTeX's stylesheet (katex/dist/katex.min.css) with its fonts as WOFF2 only. The stylesheet lists
// every font as WOFF2, WOFF and TrueType; imported as CSS, the single-file build would inline all
// three formats (about 1.1 MB), while every browser this page supports loads the WOFF2 one.
import katexCss from "katex/dist/katex.min.css?raw";
import AMSRegular from "katex/dist/fonts/KaTeX_AMS-Regular.woff2";
import CaligraphicBold from "katex/dist/fonts/KaTeX_Caligraphic-Bold.woff2";
import CaligraphicRegular from "katex/dist/fonts/KaTeX_Caligraphic-Regular.woff2";
import FrakturBold from "katex/dist/fonts/KaTeX_Fraktur-Bold.woff2";
import FrakturRegular from "katex/dist/fonts/KaTeX_Fraktur-Regular.woff2";
import MainBold from "katex/dist/fonts/KaTeX_Main-Bold.woff2";
import MainBoldItalic from "katex/dist/fonts/KaTeX_Main-BoldItalic.woff2";
import MainItalic from "katex/dist/fonts/KaTeX_Main-Italic.woff2";
import MainRegular from "katex/dist/fonts/KaTeX_Main-Regular.woff2";
import MathBoldItalic from "katex/dist/fonts/KaTeX_Math-BoldItalic.woff2";
import MathItalic from "katex/dist/fonts/KaTeX_Math-Italic.woff2";
import SansSerifBold from "katex/dist/fonts/KaTeX_SansSerif-Bold.woff2";
import SansSerifItalic from "katex/dist/fonts/KaTeX_SansSerif-Italic.woff2";
import SansSerifRegular from "katex/dist/fonts/KaTeX_SansSerif-Regular.woff2";
import ScriptRegular from "katex/dist/fonts/KaTeX_Script-Regular.woff2";
import Size1Regular from "katex/dist/fonts/KaTeX_Size1-Regular.woff2";
import Size2Regular from "katex/dist/fonts/KaTeX_Size2-Regular.woff2";
import Size3Regular from "katex/dist/fonts/KaTeX_Size3-Regular.woff2";
import Size4Regular from "katex/dist/fonts/KaTeX_Size4-Regular.woff2";
import TypewriterRegular from "katex/dist/fonts/KaTeX_Typewriter-Regular.woff2";

const FONTS: Readonly<Record<string, string>> = {
  "KaTeX_AMS-Regular": AMSRegular,
  "KaTeX_Caligraphic-Bold": CaligraphicBold,
  "KaTeX_Caligraphic-Regular": CaligraphicRegular,
  "KaTeX_Fraktur-Bold": FrakturBold,
  "KaTeX_Fraktur-Regular": FrakturRegular,
  "KaTeX_Main-Bold": MainBold,
  "KaTeX_Main-BoldItalic": MainBoldItalic,
  "KaTeX_Main-Italic": MainItalic,
  "KaTeX_Main-Regular": MainRegular,
  "KaTeX_Math-BoldItalic": MathBoldItalic,
  "KaTeX_Math-Italic": MathItalic,
  "KaTeX_SansSerif-Bold": SansSerifBold,
  "KaTeX_SansSerif-Italic": SansSerifItalic,
  "KaTeX_SansSerif-Regular": SansSerifRegular,
  "KaTeX_Script-Regular": ScriptRegular,
  "KaTeX_Size1-Regular": Size1Regular,
  "KaTeX_Size2-Regular": Size2Regular,
  "KaTeX_Size3-Regular": Size3Regular,
  "KaTeX_Size4-Regular": Size4Regular,
  "KaTeX_Typewriter-Regular": TypewriterRegular,
};

// `src:url(fonts/<name>.woff2) format("woff2"),url(fonts/<name>.woff) ...` up to the end of the
// declaration. A font this table lacks keeps its declaration (and falls back to a system font).
const FONT_SRC = /src:\s*url\((["']?)fonts\/(KaTeX_[\w-]+)\.woff2\1\)[^;}]*/g;

let installed = false;

/** Adds KaTeX's stylesheet to the document once. */
export function installKatexStyle(): void {
  if (installed || typeof document === "undefined") return;
  installed = true;
  const style = document.createElement("style");
  style.dataset.source = "katex";
  style.textContent = katexCss.replace(FONT_SRC, (match: string, _quote: string, name: string) => {
    const url = FONTS[name];
    return url ? `src:url("${url}") format("woff2")` : match;
  });
  document.head.appendChild(style);
}
