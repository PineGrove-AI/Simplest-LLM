// Minimal type declarations for @andypf/json-viewer (ships without .d.ts).
declare module "@andypf/json-viewer/dist/esm/react/JsonViewer" {
  import * as React from "react";

  export interface JsonViewerProps {
    /** The JSON value to display (object, array, or stringified JSON / URL). */
    data?: unknown;
    /** Initial depth to expand: number, true (all), or false (none). Default 1. */
    expanded?: number | boolean;
    /** Theme name (e.g. "default-dark") or a base16 CSS-variable object. */
    theme?: string | Record<string, string>;
    /** Show data-type labels next to values. Default true. */
    showDataTypes?: boolean;
    /** Show the toolbar (expand/collapse/indent/search). Default false. */
    showToolbar?: boolean;
    /** Expand icon style. Default "arrow". */
    expandIconType?: "arrow" | "square" | "circle";
    /** Show per-node copy buttons. Default true. */
    showCopy?: boolean;
    /** Show item counts on arrays/objects. Default true. */
    showSize?: boolean;
    /** Preserve expand state across data updates. Default false. */
    preserveExpanded?: boolean;
    /** Auto-expand empty objects/arrays. Default true. */
    expandEmpty?: boolean;
    /** Copy key+value instead of just value. Default false. */
    copyWithKey?: boolean;
    /** Indent width in spaces. Default 2. */
    indent?: number;
  }

  const JsonViewer: React.FC<JsonViewerProps>;
  export default JsonViewer;
}
