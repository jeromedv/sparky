import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";

/**
 * Wraps each scene with enter/exit slide transitions.
 * - Enter: opacity 0 + translateX(30px) → normal over 20 frames
 * - Exit: opacity 1 → 0 + translateX(0 → -30px) over last 15 frames
 */
export const SceneTransition: React.FC<{
  children: React.ReactNode;
  durationInFrames: number;
  enterFrames?: number;
  exitFrames?: number;
}> = ({ children, durationInFrames, enterFrames = 20, exitFrames = 15 }) => {
  const frame = useCurrentFrame();

  // Enter
  const enterOpacity = interpolate(frame, [0, enterFrames], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const enterX = interpolate(frame, [0, enterFrames], [30, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Exit
  const exitStart = durationInFrames - exitFrames;
  const exitOpacity = interpolate(frame, [exitStart, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const exitX = interpolate(frame, [exitStart, durationInFrames], [0, -30], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const opacity = Math.min(enterOpacity, exitOpacity);
  const translateX = frame <= enterFrames ? enterX : frame >= exitStart ? exitX : 0;

  return (
    <AbsoluteFill
      style={{
        opacity,
        transform: `translateX(${translateX}px)`,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};
