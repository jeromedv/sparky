import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, FONT, gridStyle } from "../styles";

export const Scene1_LogoIntro: React.FC = () => {
  const frame = useCurrentFrame();

  const logoOpacity = interpolate(frame, [0, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const logoY = interpolate(frame, [0, 25], [12, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const lineWidth = interpolate(frame, [20, 45], [0, 320], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at 50% 50%, rgba(37,99,235,0.12) 0%, transparent 70%), ${C.bg}`,
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
          opacity: logoOpacity,
          transform: `translateY(${logoY}px)`,
          fontSize: 80,
          fontWeight: 800,
          fontFamily: FONT,
          color: C.text1,
        }}
      >
        The Augmented CFO
      </div>
      <div
        style={{
          width: lineWidth,
          height: 2,
          backgroundColor: C.blue,
          borderRadius: 1,
          marginTop: 16,
        }}
      />
    </AbsoluteFill>
  );
};
