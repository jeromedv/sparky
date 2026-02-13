import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle } from "../styles";

// Scene 10 — How It Works (duration 270 frames)
// Title: local 10
// Step 1: local 40, title 50, sub 60, connector 94–130
// Step 2: local 130, connector 174–210
// Step 3: local 210, badge 216
// Exit: local 254–269

interface Step {
  num: string;
  circleColor: string;
  title: React.ReactNode;
  subtitle: React.ReactNode;
  startFrame: number;
  badge?: { text: string; color: string; frame: number };
}

const steps: Step[] = [
  {
    num: "1",
    circleColor: C.blue,
    title: "We map your highest-friction workflows",
    subtitle: "A focused audit of your current finance processes",
    startFrame: 40,
  },
  {
    num: "2",
    circleColor: C.blue,
    title: (
      <>
        We build the <span style={{ color: C.blue }}>automations</span>
      </>
    ),
    subtitle: (
      <>
        Integrated directly into your{" "}
        <span style={{ fontWeight: 700, color: C.blue }}>existing stack</span> —
        no migrations
      </>
    ),
    startFrame: 130,
  },
  {
    num: "3",
    circleColor: C.green,
    title: "First automation live by Week 2",
    subtitle: "You own everything. Full documentation. No vendor lock-in.",
    startFrame: 210,
    badge: { text: "Week 2", color: C.green, frame: 216 },
  },
];

export const Scene10_HowItWorks: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const exit = interpolate(frame, [254, 269], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Title
  const titleOp = interpolate(frame, [10, 28], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const titleY = interpolate(frame, [10, 28], [16, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: C.bg,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "50px 140px",
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          opacity: exit,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
          flex: 1,
        }}
      >
        {/* Title */}
        <div
          style={{
            opacity: titleOp,
            transform: `translateY(${titleY}px)`,
            fontSize: 38,
            fontWeight: 800,
            fontFamily: FONT,
            color: C.text1,
            marginBottom: 40,
          }}
        >
          How it works
        </div>

        {/* Steps */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            width: "100%",
            maxWidth: 900,
            flex: 1,
            justifyContent: "center",
          }}
        >
          {steps.map((step, i) => {
            const local = frame - step.startFrame;

            const stepOp = interpolate(local, [0, 18], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const stepX = interpolate(local, [0, 18], [-40, 0], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });

            const circleScale = spring({
              fps,
              frame: Math.max(0, local),
              config: { stiffness: 220, damping: 14 },
            });

            const titleOp2 = interpolate(local, [10, 24], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });
            const subOp = interpolate(local, [20, 34], [0, 1], {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
            });

            // Connector line grows to next step
            const connectorH =
              i < steps.length - 1
                ? interpolate(
                    frame,
                    [step.startFrame + 54, steps[i + 1].startFrame],
                    [0, 1],
                    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
                  )
                : 0;

            // Badge
            const badgeScale = step.badge
              ? spring({
                  fps,
                  frame: Math.max(0, frame - step.badge.frame),
                  config: { stiffness: 260, damping: 10 },
                })
              : 0;
            const badgeOp = step.badge
              ? interpolate(frame, [step.badge.frame, step.badge.frame + 8], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                })
              : 0;

            return (
              <div key={i} style={{ opacity: stepOp }}>
                <div
                  style={{
                    transform: `translateX(${stepX}px)`,
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 24,
                  }}
                >
                  {/* Circle */}
                  <div
                    style={{
                      transform: `scale(${circleScale})`,
                      width: 44,
                      height: 44,
                      borderRadius: 22,
                      backgroundColor: step.circleColor,
                      display: "flex",
                      justifyContent: "center",
                      alignItems: "center",
                      fontSize: 20,
                      fontWeight: 800,
                      fontFamily: FONT,
                      color: "#FFFFFF",
                      flexShrink: 0,
                    }}
                  >
                    {step.num}
                  </div>

                  {/* Content */}
                  <div style={{ flex: 1, paddingTop: 2 }}>
                    <div
                      style={{
                        opacity: titleOp2,
                        display: "flex",
                        alignItems: "center",
                        gap: 12,
                        marginBottom: 6,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 20,
                          fontWeight: 700,
                          fontFamily: FONT,
                          color: C.text1,
                        }}
                      >
                        {step.title}
                      </span>
                      {step.badge && (
                        <span
                          style={{
                            opacity: badgeOp,
                            transform: `scale(${badgeScale})`,
                            display: "inline-block",
                            backgroundColor: `rgba(16,185,129,0.15)`,
                            border: `1px solid rgba(16,185,129,0.40)`,
                            borderRadius: 16,
                            padding: "4px 14px",
                            fontSize: 13,
                            fontWeight: 700,
                            fontFamily: FONT,
                            color: step.badge.color,
                          }}
                        >
                          {step.badge.text}
                        </span>
                      )}
                    </div>
                    <div
                      style={{
                        opacity: subOp,
                        fontSize: 14,
                        fontWeight: 400,
                        fontFamily: FONT,
                        color: C.text2,
                      }}
                    >
                      {step.subtitle}
                    </div>
                  </div>
                </div>

                {/* Connector line */}
                {i < steps.length - 1 && (
                  <div
                    style={{
                      marginLeft: 21,
                      width: 2,
                      height: 40,
                      backgroundColor: C.border,
                      transform: `scaleY(${connectorH})`,
                      transformOrigin: "top",
                      marginTop: 6,
                      marginBottom: 6,
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    </AbsoluteFill>
  );
};
