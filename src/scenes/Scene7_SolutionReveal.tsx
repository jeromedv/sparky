import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  interpolateColors,
  spring,
} from "remotion";
import { C, FONT, gridStyle, glowBlueBright } from "../styles";

export const Scene7_SolutionReveal: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const badgeScale = spring({
    fps,
    frame: Math.max(0, frame - 15),
    config: { stiffness: 200, damping: 14 },
  });
  const badgeOp = interpolate(frame, [15, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const l1Op = interpolate(frame, [40, 58], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l1Y = interpolate(frame, [40, 58], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const aiColor = interpolateColors(frame, [50, 60], [C.text2, C.green]);

  // Line 2 — "So your team can focus…" fades in 20 frames after line 1
  const l2Op = interpolate(frame, [60, 78], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l2Y = interpolate(frame, [60, 78], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 3 phrases — 15-frame stagger
  const p1Op = interpolate(frame, [100, 112], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const p2Op = interpolate(frame, [115, 127], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const p3Op = interpolate(frame, [130, 142], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `${glowBlueBright}, ${C.bg}`,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        gap: 20,
        padding: 80,
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 20,
        }}
      >
        <div
          style={{
            opacity: badgeOp,
            transform: `scale(${badgeScale})`,
            border: "1px solid rgba(37,99,235,0.6)",
            backgroundColor: "rgba(37,99,235,0.10)",
            borderRadius: 30,
            padding: "10px 24px",
            marginBottom: 20,
          }}
        >
          <span
            style={{
              fontSize: 32,
              fontWeight: 700,
              fontFamily: FONT,
              color: C.text1,
            }}
          >
            The Augmented CFO
          </span>
        </div>

        {/* Line 1 */}
        <div
          style={{
            opacity: l1Op,
            transform: `translateY(${l1Y}px)`,
            fontSize: 52,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
            textAlign: "center",
          }}
        >
          We build <span style={{ color: aiColor }}>AI automations</span>{" "}
          directly into your finance stack.
        </div>

        {/* Line 2 */}
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 40,
            fontWeight: 600,
            fontFamily: FONT,
            color: "#CBD5E1",
            textAlign: "center",
          }}
        >
          So your team can focus on what actually matters.
        </div>

        {/* Line 3 */}
        <div
          style={{
            display: "flex",
            gap: 32,
            marginTop: 24,
          }}
        >
          <span
            style={{
              opacity: p1Op,
              fontSize: 32,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#CBD5E1",
            }}
          >
            No new hires.
          </span>
          <span
            style={{
              opacity: p2Op,
              fontSize: 32,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#CBD5E1",
            }}
          >
            No coding.
          </span>
          <span
            style={{
              opacity: p3Op,
              fontSize: 32,
              fontWeight: 500,
              fontFamily: FONT,
              color: "#CBD5E1",
            }}
          >
            No 6-month projects.
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
