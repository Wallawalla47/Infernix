// The categorical palette the charts share (the Usage page's models, the dashboard's memory parts).

import { useTheme, type ResolvedTheme } from "./theme";

/** The validated categorical palette in its fixed order: slots 1-3 are the theme's series tokens,
 * 4-8 the same palette's further hues stepped per theme. A chart never generates a ninth hue: it
 * folds the rest into "Other" or tells them apart another way (a pattern, a neutral). */
export const CATEGORICAL: Record<ResolvedTheme, readonly string[]> = {
  light: ["var(--series-1)", "var(--series-2)", "var(--series-3)", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["var(--series-1)", "var(--series-2)", "var(--series-3)", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};

/** A series without its own slot ("Other"): the theme's muted chart grey. */
export const OTHER_COLOR = "var(--chart-muted)";

/** The categorical palette of the theme in use. */
export function useCategoricalPalette(): readonly string[] {
  return CATEGORICAL[useTheme().resolved];
}
