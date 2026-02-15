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

  const l3Op = interpolate(frame, [34, 54], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const l3Y = interpolate(frame, [34, 54], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const underlineX = interpolate(frame, [36, 59], [0, 1], {
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
          You have 3 days
        </div>
        <div
          style={{
            opacity: l2Op,
            transform: `translateY(${l2Y}px)`,
            fontSize: 80,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
          }}
        >
          to close the books.
        </div>
        <div
          style={{
            opacity: l3Op,
            transform: `translateY(${l3Y}px)`,
            fontSize: 80,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.red,
            position: "relative",
            display: "inline-block",
          }}
        >
          Where do things stand?
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
