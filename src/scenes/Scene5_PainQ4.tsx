import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { C, FONT, gridStyle, glowRed } from "../styles";

export const Scene5_PainQ4: React.FC = () => {
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

  const strikeX = interpolate(frame, [39, 59], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const l3Op = interpolate(frame, [64, 80], [0, 1], {
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
            fontSize: 80,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          Your best people
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 80,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
            display: "flex",
            gap: 16,
          }}
        >
          <span>are doing</span>
          <span style={{ position: "relative", display: "inline-block" }}>
            copy-paste.
            <div
              style={{
                position: "absolute",
                top: "55%",
                left: 0,
                width: "100%",
                height: 4,
                backgroundColor: C.red,
                transform: `scaleX(${strikeX})`,
                transformOrigin: "left center",
                borderRadius: 2,
              }}
            />
          </span>
        </div>
        <div
          style={{
            opacity: l3Op,
            fontSize: 52,
            fontWeight: 600,
            fontFamily: FONT,
            color: C.text1,
            marginTop: 16,
          }}
        >
          That's not analysis. That's waste.
        </div>
      </div>
    </AbsoluteFill>
  );
};
