import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle, glowRed } from "../styles";

export const Scene4_PainQ3: React.FC = () => {
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

  // 20-frame gap before line 2
  const l2Op = interpolate(frame, [30, 50], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l2Y = interpolate(frame, [30, 50], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const emojiScale = spring({
    fps,
    frame: Math.max(0, frame - 34),
    config: { stiffness: 220, damping: 14 },
  });
  const emojiOp = interpolate(frame, [34, 40], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
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
            fontSize: 58,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          Your board wants answers.
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 58,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}
        >
          You have exports.
          <span
            style={{
              opacity: emojiOp,
              transform: `scale(${emojiScale})`,
              display: "inline-block",
              fontSize: 48,
            }}
          >
            📊
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
