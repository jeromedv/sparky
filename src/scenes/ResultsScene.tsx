import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

export const ResultsScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Animated counter for hours saved
  const hoursMin = interpolate(frame, [20, 55], [0, 20], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const hoursMax = interpolate(frame, [20, 55], [0, 40], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Big number reveal
  const numberScale = spring({
    fps,
    frame: Math.max(0, frame - 15),
    config: { damping: 80, stiffness: 150 },
  });

  const numberOpacity = interpolate(frame, [15, 25], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Bottom text
  const bottomOpacity = interpolate(frame, [50, 65], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const bottomY = interpolate(frame, [50, 65], [20, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Glow pulse
  const glowIntensity = interpolate(
    frame,
    [20, 45, 70, 90],
    [0.1, 0.4, 0.3, 0.4],
    {
      extrapolateLeft: "clamp",
      extrapolateRight: "clamp",
    }
  );

  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(ellipse at center, rgba(16, 185, 129, ${glowIntensity}) 0%, #0a0a1a 65%)`,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: 80,
      }}
    >
      {/* Label */}
      <div
        style={{
          opacity: titleOpacity,
          fontSize: 24,
          fontFamily: "sans-serif",
          color: "#34d399",
          letterSpacing: 4,
          textTransform: "uppercase",
          marginBottom: 20,
          fontWeight: 600,
        }}
      >
        The Results
      </div>

      {/* Big number */}
      <div
        style={{
          opacity: numberOpacity,
          transform: `scale(${numberScale})`,
          textAlign: "center",
          marginBottom: 10,
        }}
      >
        <div
          style={{
            fontSize: 160,
            fontWeight: 900,
            fontFamily: "sans-serif",
            background:
              "linear-gradient(135deg, #34d399 0%, #60a5fa 50%, #a78bfa 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            lineHeight: 1,
          }}
        >
          {Math.round(hoursMin)}-{Math.round(hoursMax)}h
        </div>
        <div
          style={{
            fontSize: 42,
            fontWeight: 700,
            fontFamily: "sans-serif",
            color: "#ffffff",
            marginTop: 10,
          }}
        >
          saved per month
        </div>
      </div>

      {/* Bottom description */}
      <div
        style={{
          opacity: bottomOpacity,
          transform: `translateY(${bottomY}px)`,
          textAlign: "center",
          marginTop: 40,
          maxWidth: 800,
        }}
      >
        <div
          style={{
            fontSize: 28,
            fontFamily: "sans-serif",
            color: "#94a3b8",
            lineHeight: 1.5,
          }}
        >
          Through <span style={{ color: "#34d399", fontWeight: 600 }}>AI</span>{" "}
          and{" "}
          <span style={{ color: "#60a5fa", fontWeight: 600 }}>automation</span>,
          we eliminate the busywork so you can focus on{" "}
          <span style={{ color: "#ffffff", fontWeight: 600 }}>
            strategic decisions
          </span>
        </div>
      </div>
    </AbsoluteFill>
  );
};
