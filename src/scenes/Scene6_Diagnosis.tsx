import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle, glowBlue } from "../styles";

export const Scene6_Diagnosis: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const exit = interpolate(frame, [134, 149], [1, 0], {
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

  // Divider — local 54
  const dividerX = interpolate(frame, [54, 74], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 3 — local 69
  const l3Op = interpolate(frame, [69, 89], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l3Y = interpolate(frame, [69, 89], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const weFixScale = spring({
    fps,
    frame: Math.max(0, frame - 69),
    config: { stiffness: 140, damping: 16 },
  });

  return (
    <AbsoluteFill
      style={{
        background: `${glowBlue}, ${C.bg}`,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          opacity: exit,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 16,
        }}
      >
        <div
          style={{
            opacity: l1Op,
            transform: `translateY(${l1Y}px)`,
            fontSize: 44,
            fontWeight: 600,
            fontFamily: FONT,
            color: "#E2E8F0",
          }}
        >
          This isn't a skills problem.
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 44,
            fontWeight: 600,
            fontFamily: FONT,
            color: "#E2E8F0",
          }}
        >
          It's a process problem.
        </div>
        <div
          style={{
            width: 120,
            height: 1,
            backgroundColor: C.border,
            transform: `scaleX(${dividerX})`,
            marginTop: 8,
            marginBottom: 8,
          }}
        />
        <div
          style={{
            opacity: l3Op,
            transform: `translateY(${l3Y}px) scale(${weFixScale})`,
            fontSize: 44,
            fontWeight: 800,
            fontFamily: FONT,
          }}
        >
          <span style={{ color: C.text1 }}>And that's exactly what </span>
          <span style={{ color: C.blue }}>we fix.</span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
