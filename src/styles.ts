import React from "react";
import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

// Brand Tokens — Dark Premium
export const C = {
  bg: "#0A0F1E",
  surface: "#111827",
  border: "#1F2937",
  blue: "#2563EB",
  green: "#10B981",
  red: "#EF4444",
  gold: "#F59E0B",
  text1: "#F9FAFB",
  text2: "#9CA3AF",
  text3: "#4B5563",
};

export const FONT = fontFamily;

// Glow presets
export const glowRed =
  "radial-gradient(ellipse at 50% 40%, rgba(239,68,68,0.08) 0%, transparent 60%)";
export const glowBlue =
  "radial-gradient(ellipse at 50% 50%, rgba(37,99,235,0.10) 0%, transparent 65%)";
export const glowBlueBright =
  "radial-gradient(ellipse at 50% 45%, rgba(37,99,235,0.20) 0%, transparent 60%)";
export const glowGreen =
  "radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.08) 0%, transparent 65%)";
export const glowBlueCTA =
  "radial-gradient(ellipse at 50% 45%, rgba(37,99,235,0.18) 0%, transparent 60%)";

// Grid overlay
export const gridStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  backgroundImage:
    "linear-gradient(#1F293720 1px, transparent 1px), linear-gradient(90deg, #1F293720 1px, transparent 1px)",
  backgroundSize: "60px 60px",
  opacity: 0.4,
  pointerEvents: "none",
};
