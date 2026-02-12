import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

const painPoints = [
  { icon: "🕐", text: "Manual data entry & reconciliation" },
  { icon: "📊", text: "Repetitive reporting tasks" },
  { icon: "🔄", text: "Disconnected financial workflows" },
  { icon: "⚠️", text: "Slow month-end close processes" },
];

export const ProblemScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleX = interpolate(frame, [0, 15], [-50, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(ellipse at top right, rgba(239, 68, 68, 0.15) 0%, #0a0a1a 60%)",
        display: "flex",
        flexDirection: "row",
        padding: 80,
        alignItems: "center",
      }}
    >
      {/* Left side - Title */}
      <div
        style={{
          flex: 1,
          opacity: titleOpacity,
          transform: `translateX(${titleX}px)`,
        }}
      >
        <div
          style={{
            fontSize: 24,
            fontFamily: "sans-serif",
            color: "#f87171",
            letterSpacing: 4,
            textTransform: "uppercase",
            marginBottom: 20,
            fontWeight: 600,
          }}
        >
          The Problem
        </div>
        <div
          style={{
            fontSize: 56,
            fontWeight: 800,
            fontFamily: "sans-serif",
            color: "#ffffff",
            lineHeight: 1.2,
          }}
        >
          Finance teams are
          <br />
          <span style={{ color: "#f87171" }}>drowning</span> in
          <br />
          manual work
        </div>
      </div>

      {/* Right side - Pain points */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 24,
          paddingLeft: 60,
        }}
      >
        {painPoints.map((point, index) => {
          const delay = 15 + index * 12;
          const itemOpacity = interpolate(frame, [delay, delay + 10], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const itemX = interpolate(frame, [delay, delay + 10], [40, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          return (
            <div
              key={index}
              style={{
                opacity: itemOpacity,
                transform: `translateX(${itemX}px)`,
                display: "flex",
                alignItems: "center",
                gap: 20,
                backgroundColor: "rgba(239, 68, 68, 0.08)",
                border: "1px solid rgba(239, 68, 68, 0.2)",
                borderRadius: 16,
                padding: "20px 28px",
              }}
            >
              <span style={{ fontSize: 36 }}>{point.icon}</span>
              <span
                style={{
                  fontSize: 26,
                  fontFamily: "sans-serif",
                  color: "#e2e8f0",
                  fontWeight: 500,
                }}
              >
                {point.text}
              </span>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
