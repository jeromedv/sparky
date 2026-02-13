import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, FONT, gridStyle, glowGreen } from "../styles";

const automatedItems = [
  "Reconciliations",
  "Variance commentary",
  "Data consolidation",
  "Close tracking",
  "Report assembly",
];

const teamItems = [
  "Strategic analysis",
  "Board advisory",
  "Business partnering",
  "Scenario planning",
  "Decisions that move the needle",
];

export const Scene9_HighValueWork: React.FC = () => {
  const frame = useCurrentFrame();

  const exit = interpolate(frame, [164, 179], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 1 — local 20
  const l1Op = interpolate(frame, [20, 38], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l1Y = interpolate(frame, [20, 38], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 2 — local 40
  const l2Op = interpolate(frame, [40, 58], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l2Y = interpolate(frame, [40, 58], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Card A slides in from left — local 70
  const cardAOp = interpolate(frame, [70, 90], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cardAX = interpolate(frame, [70, 90], [-60, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Card B slides in from right — local 80
  const cardBOp = interpolate(frame, [80, 100], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cardBX = interpolate(frame, [80, 100], [60, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `${glowGreen}, ${C.bg}`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "60px 100px",
        gap: 32,
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          opacity: exit,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 32,
          width: "100%",
        }}
      >
        {/* Text */}
        <div style={{ textAlign: "center" }}>
          <div
            style={{
              opacity: l1Op,
              transform: `translateY(${l1Y}px)`,
              fontSize: 34,
              fontWeight: 600,
              fontFamily: FONT,
              color: C.text2,
              marginBottom: 12,
            }}
          >
            AI handles the repetitive work.
          </div>
          <div
            style={{
              opacity: l2Op,
              transform: `translateY(${l2Y}px)`,
              fontSize: 40,
              fontWeight: 800,
              fontFamily: FONT,
            }}
          >
            <span style={{ color: C.text1 }}>Your team focuses on </span>
            <span style={{ color: C.green }}>what matters.</span>
          </div>
        </div>

        {/* Cards */}
        <div style={{ display: "flex", gap: 28, width: "100%", maxWidth: 1000 }}>
          {/* Card A — Automated away */}
          <div
            style={{
              opacity: cardAOp,
              transform: `translateX(${cardAX}px)`,
              flex: 1,
              backgroundColor: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.25)",
              borderRadius: 12,
              padding: 20,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 600,
                fontFamily: FONT,
                color: C.red,
                marginBottom: 16,
              }}
            >
              🔁{"  "}Automated away
            </div>
            {automatedItems.map((item, i) => {
              const d = 90 + i * 8;
              const iOp = interpolate(frame, [d, d + 10], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              });
              const iX = interpolate(frame, [d, d + 10], [-10, 0], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              });
              return (
                <div
                  key={i}
                  style={{
                    opacity: iOp,
                    transform: `translateX(${iX}px)`,
                    fontSize: 13,
                    fontWeight: 400,
                    fontFamily: FONT,
                    color: C.text2,
                    padding: "6px 0",
                  }}
                >
                  {item}
                </div>
              );
            })}
          </div>

          {/* Card B — Your team */}
          <div
            style={{
              opacity: cardBOp,
              transform: `translateX(${cardBX}px)`,
              flex: 1,
              backgroundColor: "rgba(16,185,129,0.08)",
              border: "1px solid rgba(16,185,129,0.25)",
              borderRadius: 12,
              padding: 20,
            }}
          >
            <div
              style={{
                fontSize: 14,
                fontWeight: 600,
                fontFamily: FONT,
                color: C.green,
                marginBottom: 16,
              }}
            >
              🧠{"  "}Your team
            </div>
            {teamItems.map((item, i) => {
              const d = 100 + i * 8;
              const iOp = interpolate(frame, [d, d + 10], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              });
              const iX = interpolate(frame, [d, d + 10], [-10, 0], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              });
              return (
                <div
                  key={i}
                  style={{
                    opacity: iOp,
                    transform: `translateX(${iX}px)`,
                    fontSize: 13,
                    fontWeight: 400,
                    fontFamily: FONT,
                    color: C.text2,
                    padding: "6px 0",
                  }}
                >
                  {item}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
