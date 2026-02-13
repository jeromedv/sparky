import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
} from "remotion";
import { COLORS, FONT, centerFlex } from "../styles";

export const Scene3_Diagnosis: React.FC = () => {
  const frame = useCurrentFrame();

  // Line 1: "This isn't a skills problem."
  const line1Opacity = interpolate(frame, [10, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line1Y = interpolate(frame, [10, 28], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 2: "It's a process problem."
  const line2Opacity = interpolate(frame, [35, 53], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line2Y = interpolate(frame, [35, 53], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Divider line
  const dividerWidth = interpolate(frame, [60, 85], [0, 120], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Line 3: "And that's exactly what we fix."
  const line3Opacity = interpolate(frame, [75, 95], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const line3Y = interpolate(frame, [75, 95], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        ...centerFlex,
        backgroundColor: COLORS.lightBg,
        gap: 20,
      }}
    >
      {/* Line 1 */}
      <div
        style={{
          opacity: line1Opacity,
          transform: `translateY(${line1Y}px)`,
          fontSize: 42,
          fontWeight: 500,
          fontFamily: FONT,
          color: COLORS.secondaryText,
        }}
      >
        This isn't a skills problem.
      </div>

      {/* Line 2 */}
      <div
        style={{
          opacity: line2Opacity,
          transform: `translateY(${line2Y}px)`,
          fontSize: 42,
          fontWeight: 500,
          fontFamily: FONT,
          color: COLORS.secondaryText,
        }}
      >
        It's a process problem.
      </div>

      {/* Divider */}
      <div
        style={{
          width: dividerWidth,
          height: 2,
          backgroundColor: COLORS.blue,
          borderRadius: 1,
          marginTop: 12,
          marginBottom: 12,
        }}
      />

      {/* Line 3 */}
      <div
        style={{
          opacity: line3Opacity,
          transform: `translateY(${line3Y}px)`,
          fontSize: 52,
          fontWeight: 700,
          fontFamily: FONT,
          color: COLORS.blue,
        }}
      >
        And that's exactly what we fix.
      </div>
    </AbsoluteFill>
  );
};
