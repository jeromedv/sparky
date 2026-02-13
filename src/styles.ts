import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

// Brand Colors
export const COLORS = {
  background: "#FFFFFF",
  primaryText: "#0F172A",
  blue: "#2563EB",
  secondaryText: "#475569",
  success: "#059669",
  negative: "#DC2626",
  gold: "#D97706",
  lightBg: "#F8FAFC",
  darkBg: "#1A1A2E",
} as const;

// Typography
export const FONT = fontFamily;

export const centerFlex: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  flexDirection: "column",
};
