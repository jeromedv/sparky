import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  interpolate,
  spring,
  useVideoConfig,
} from "remotion";
import { COLORS, FONT } from "../styles";

interface Step {
  number: string;
  title: string;
  titleHighlight?: { text: string; color: string };
  subtitle: string;
  badge?: { text: string; color: string };
  delay: number;
}

const steps: Step[] = [
  {
    number: "1",
    title: "We identify your most time-consuming workflows",
    subtitle: "Audit of your current finance processes",
    delay: 25,
  },
  {
    number: "2",
    title: "We build the AI automations",
    titleHighlight: undefined,
    subtitle: "Integrated directly into your existing stack",
    delay: 115,
  },
  {
    number: "3",
    title: "First automation live by Week 2",
    subtitle: "You own everything. Full documentation. No lock-in.",
    badge: { text: "Week 2", color: COLORS.success },
    delay: 205,
  },
];

export const Scene7_HowItWorks: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title
  const titleOpacity = interpolate(frame, [0, 18], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.background,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "60px 120px",
      }}
    >
      {/* Title */}
      <div
        style={{
          opacity: titleOpacity,
          fontSize: 48,
          fontWeight: 800,
          fontFamily: FONT,
          color: COLORS.primaryText,
          marginBottom: 50,
        }}
      >
        How it works
      </div>

      {/* Steps */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 0,
          width: "100%",
          maxWidth: 1000,
          flex: 1,
          justifyContent: "center",
        }}
      >
        {steps.map((step, i) => {
          const localFrame = frame - step.delay;

          const stepOpacity = interpolate(localFrame, [0, 18], [0, 1], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });
          const stepX = interpolate(localFrame, [0, 18], [-50, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          });

          // Number circle spring
          const circleScale = spring({
            fps,
            frame: Math.max(0, localFrame),
            config: { damping: 80, stiffness: 200 },
          });

          // Connecting line grows
          const lineHeight =
            i < steps.length - 1
              ? interpolate(localFrame, [20, 70], [0, 50], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                })
              : 0;

          // Badge spring for step 3
          const badgeScale = step.badge
            ? spring({
                fps,
                frame: Math.max(0, localFrame - 22),
                config: { damping: 60, stiffness: 200 },
              })
            : 0;
          const badgeOpacity = step.badge
            ? interpolate(localFrame, [20, 30], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              })
            : 0;

          return (
            <div key={i} style={{ opacity: stepOpacity }}>
              <div
                style={{
                  transform: `translateX(${stepX}px)`,
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 28,
                }}
              >
                {/* Number circle */}
                <div
                  style={{
                    transform: `scale(${circleScale})`,
                    width: 56,
                    height: 56,
                    borderRadius: 28,
                    backgroundColor: COLORS.blue,
                    display: "flex",
                    justifyContent: "center",
                    alignItems: "center",
                    fontSize: 24,
                    fontWeight: 800,
                    fontFamily: FONT,
                    color: "#FFFFFF",
                    flexShrink: 0,
                  }}
                >
                  {step.number}
                </div>

                {/* Text content */}
                <div style={{ flex: 1, paddingTop: 4 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 16,
                      marginBottom: 8,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 30,
                        fontWeight: 700,
                        fontFamily: FONT,
                        color: COLORS.primaryText,
                      }}
                    >
                      {step.title}
                    </span>
                    {/* Badge */}
                    {step.badge && (
                      <span
                        style={{
                          opacity: badgeOpacity,
                          transform: `scale(${badgeScale})`,
                          display: "inline-block",
                          backgroundColor: step.badge.color,
                          color: "#FFFFFF",
                          fontSize: 16,
                          fontWeight: 700,
                          fontFamily: FONT,
                          padding: "6px 16px",
                          borderRadius: 20,
                        }}
                      >
                        {step.badge.text}
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      fontSize: 22,
                      fontWeight: 400,
                      fontFamily: FONT,
                      color: COLORS.secondaryText,
                    }}
                  >
                    {step.number === "2" ? (
                      <>
                        Integrated directly into your{" "}
                        <span
                          style={{ fontWeight: 700, color: COLORS.blue }}
                        >
                          existing stack
                        </span>
                      </>
                    ) : (
                      step.subtitle
                    )}
                  </div>
                </div>
              </div>

              {/* Connecting line */}
              {i < steps.length - 1 && (
                <div
                  style={{
                    marginLeft: 27,
                    width: 2,
                    height: lineHeight,
                    backgroundColor: "#E2E8F0",
                    marginTop: 8,
                    marginBottom: 8,
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
