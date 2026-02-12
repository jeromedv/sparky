import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";

const services = [
  { icon: "🤖", title: "AI Automation", desc: "Automate repetitive finance tasks" },
  { icon: "📈", title: "Smart Reporting", desc: "AI-powered financial insights" },
  { icon: "⚡", title: "Workflow Optimization", desc: "Streamlined processes end-to-end" },
];

export const SolutionScene: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const titleOpacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const titleScale = spring({
    fps,
    frame,
    config: { damping: 120 },
  });

  return (
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(ellipse at center, rgba(37, 99, 235, 0.2) 0%, #0a0a1a 70%)",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: 80,
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          transform: `scale(${titleScale})`,
          textAlign: "center",
          marginBottom: 60,
        }}
      >
        <div
          style={{
            fontSize: 24,
            fontFamily: "sans-serif",
            color: "#60a5fa",
            letterSpacing: 4,
            textTransform: "uppercase",
            marginBottom: 16,
            fontWeight: 600,
          }}
        >
          Our Solution
        </div>
        <div
          style={{
            fontSize: 52,
            fontWeight: 800,
            fontFamily: "sans-serif",
            color: "#ffffff",
            lineHeight: 1.2,
          }}
        >
          We help <span style={{ color: "#60a5fa" }}>CFOs</span>,{" "}
          <span style={{ color: "#818cf8" }}>Fractional CFOs</span>
          <br />& <span style={{ color: "#a78bfa" }}>Finance Teams</span>{" "}
          leverage AI
        </div>
      </div>

      {/* Service cards */}
      <div
        style={{
          display: "flex",
          gap: 40,
          justifyContent: "center",
        }}
      >
        {services.map((service, index) => {
          const delay = 20 + index * 10;
          const cardOpacity = interpolate(
            frame,
            [delay, delay + 12],
            [0, 1],
            {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            }
          );
          const cardY = interpolate(frame, [delay, delay + 12], [40, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          return (
            <div
              key={index}
              style={{
                opacity: cardOpacity,
                transform: `translateY(${cardY}px)`,
                backgroundColor: "rgba(59, 130, 246, 0.08)",
                border: "1px solid rgba(59, 130, 246, 0.25)",
                borderRadius: 20,
                padding: "40px 36px",
                width: 320,
                textAlign: "center",
              }}
            >
              <div style={{ fontSize: 48, marginBottom: 16 }}>
                {service.icon}
              </div>
              <div
                style={{
                  fontSize: 28,
                  fontWeight: 700,
                  fontFamily: "sans-serif",
                  color: "#ffffff",
                  marginBottom: 10,
                }}
              >
                {service.title}
              </div>
              <div
                style={{
                  fontSize: 20,
                  fontFamily: "sans-serif",
                  color: "#94a3b8",
                  fontWeight: 400,
                }}
              >
                {service.desc}
              </div>
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
