import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
} from "remotion";
import { COLORS, FONT } from "../styles";

const automatedItems = [
  "Reconciliations",
  "Variance commentary",
  "Data consolidation",
  "Close tracking",
];

const teamItems = [
  "Strategic analysis",
  "Informed decisions",
  "Board advisory",
  "Business partnering",
];

export const Scene6_HighValueWork: React.FC = () => {
  const frame = useCurrentFrame();

  // Left side text
  const line1Opacity = interpolate(frame, [8, 24], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line1Y = interpolate(frame, [8, 24], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const line2Opacity = interpolate(frame, [24, 42], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line2Y = interpolate(frame, [24, 42], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Cards appear
  const cardAOpacity = interpolate(frame, [35, 52], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cardAX = interpolate(frame, [35, 52], [-40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const cardBOpacity = interpolate(frame, [50, 67], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cardBX = interpolate(frame, [50, 67], [40, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.background,
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        padding: "60px 80px",
        gap: 60,
      }}
    >
      {/* Left column - text */}
      <div style={{ flex: 1 }}>
        <div
          style={{
            opacity: line1Opacity,
            transform: `translateY(${line1Y}px)`,
            fontSize: 32,
            fontWeight: 400,
            fontFamily: FONT,
            color: COLORS.secondaryText,
            marginBottom: 16,
            lineHeight: 1.4,
          }}
        >
          AI handles the repetitive work.
        </div>
        <div
          style={{
            opacity: line2Opacity,
            transform: `translateY(${line2Y}px)`,
            fontSize: 44,
            fontWeight: 800,
            fontFamily: FONT,
            color: COLORS.primaryText,
            lineHeight: 1.3,
          }}
        >
          Your team focuses on what actually moves the business.
        </div>
      </div>

      {/* Right column - cards */}
      <div style={{ flex: 1, display: "flex", gap: 24 }}>
        {/* Card A - Automated by AI (red tones) */}
        <div
          style={{
            opacity: cardAOpacity,
            transform: `translateX(${cardAX}px)`,
            backgroundColor: "#FEF2F2",
            borderRadius: 20,
            padding: "32px 28px",
            flex: 1,
            border: "1px solid #FECACA",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              fontFamily: FONT,
              color: COLORS.primaryText,
              marginBottom: 20,
            }}
          >
            🔁 Automated by AI
          </div>
          {automatedItems.map((item, i) => {
            const itemDelay = 55 + i * 8;
            const itemOpacity = interpolate(
              frame,
              [itemDelay, itemDelay + 10],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            return (
              <div
                key={i}
                style={{
                  opacity: itemOpacity,
                  fontSize: 20,
                  fontWeight: 500,
                  fontFamily: FONT,
                  color: COLORS.secondaryText,
                  padding: "8px 0",
                  borderBottom:
                    i < automatedItems.length - 1
                      ? "1px solid #FEE2E2"
                      : "none",
                }}
              >
                {item}
              </div>
            );
          })}
        </div>

        {/* Card B - Your Team (green tones) */}
        <div
          style={{
            opacity: cardBOpacity,
            transform: `translateX(${cardBX}px)`,
            backgroundColor: "#F0FDF4",
            borderRadius: 20,
            padding: "32px 28px",
            flex: 1,
            border: "1px solid #BBF7D0",
          }}
        >
          <div
            style={{
              fontSize: 22,
              fontWeight: 700,
              fontFamily: FONT,
              color: COLORS.primaryText,
              marginBottom: 20,
            }}
          >
            🧠 Your Team
          </div>
          {teamItems.map((item, i) => {
            const itemDelay = 70 + i * 8;
            const itemOpacity = interpolate(
              frame,
              [itemDelay, itemDelay + 10],
              [0, 1],
              { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
            );
            return (
              <div
                key={i}
                style={{
                  opacity: itemOpacity,
                  fontSize: 20,
                  fontWeight: 500,
                  fontFamily: FONT,
                  color: COLORS.secondaryText,
                  padding: "8px 0",
                  borderBottom:
                    i < teamItems.length - 1
                      ? "1px solid #D1FAE5"
                      : "none",
                }}
              >
                {item}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
