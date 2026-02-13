import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle, glowRed } from "../styles";

export const Scene2_PainQ1: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const exit = interpolate(frame, [104, 119], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const l1Op = interpolate(frame, [10, 30], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l1Y = interpolate(frame, [10, 30], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const l2Op = interpolate(frame, [22, 42], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l2Y = interpolate(frame, [22, 42], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const l3Op = interpolate(frame, [34, 54], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l3Y = interpolate(frame, [34, 54], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const closeScale = spring({
    fps,
    frame: Math.max(0, frame - 34),
    config: { stiffness: 180, damping: 14 },
  });

  return (
    <AbsoluteFill
      style={{
        background: `${glowRed}, ${C.bg}`,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column",
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          opacity: exit,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 8,
        }}
      >
        <div
          style={{
            opacity: l1Op,
            transform: `translateY(${l1Y}px)`,
            fontSize: 64,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          How many hours
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 64,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          does your team lose
        </div>
        <div
          style={{
            opacity: l3Op,
            transform: `translateY(${l3Y}px)`,
            fontSize: 64,
            fontWeight: 800,
            fontFamily: FONT,
            display: "flex",
            gap: 16,
          }}
        >
          <span style={{ color: C.text1 }}>every month</span>
          <span
            style={{
              color: C.red,
              transform: `scale(${closeScale})`,
              display: "inline-block",
            }}
          >
            on the close?
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
