import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, FONT, gridStyle, glowRed } from "../styles";

export const Scene3_PainQ2: React.FC = () => {
  const frame = useCurrentFrame();

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

  const underlineX = interpolate(frame, [36, 56], [0, 1], {
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
        padding: 80,
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
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
          Your board wants answers by Friday.
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 64,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.red,
            position: "relative",
            display: "inline-block",
          }}
        >
          You're still consolidating data.
          <div
            style={{
              position: "absolute",
              bottom: -4,
              left: 0,
              width: "100%",
              height: 3,
              backgroundColor: C.red,
              transform: `scaleX(${underlineX})`,
              transformOrigin: "left",
            }}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};
