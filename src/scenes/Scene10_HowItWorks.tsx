import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { C, FONT, gridStyle } from "../styles";

interface Step {
  num: string;
  circleColor: string;
  title: React.ReactNode;
  subtitle: React.ReactNode;
  startFrame: number;
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
        We build the <span style={{ color: C.green }}>automations</span>
      </>
    ),
    subtitle: (
      <>
        Integrated directly into your{" "}
        <span style={{ fontWeight: 700, color: C.green }}>existing stack</span>,
        no migrations required.
      </>
    ),
    startFrame: 130,
  },
  {
    num: "3",
    circleColor: C.green,
    title: (
      <>
        First automation live by{" "}
        <span style={{ color: C.green }}>Week 2</span>
      </>
    ),
    subtitle: "You own everything. Full documentation. No vendor lock-in.",
    startFrame: 210,
  },
];

export const Scene10_HowItWorks: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

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
        justifyContent: "center",
        padding: 80,
      }}
    >
      <div style={gridStyle} />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          width: "100%",
        }}
      >
        {/* Title */}
        <div
          style={{
            opacity: titleOp,
            transform: `translateY(${titleY}px)`,
            fontSize: 80,
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
                  {/* Circle — 56px diameter */}
                  <div
                    style={{
                      transform: `scale(${circleScale})`,
                      width: 56,
                      height: 56,
                      borderRadius: 28,
                      backgroundColor: step.circleColor,
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
                    {step.num}
                  </div>

                  {/* Content */}
                  <div style={{ flex: 1, paddingTop: 4 }}>
                    <div
                      style={{
                        opacity: titleOp2,
                        marginBottom: 8,
                      }}
                    >
                      <span
                        style={{
                          fontSize: 32,
                          fontWeight: 700,
                          fontFamily: FONT,
                          color: C.text1,
                        }}
                      >
                        {step.title}
                      </span>
                    </div>
                    <div
                      style={{
                        opacity: subOp,
                        fontSize: 30,
                        fontWeight: 500,
                        fontFamily: FONT,
                        color: "#CBD5E1",
                      }}
                    >
                      {step.subtitle}
                    </div>
                  </div>
                </div>

                {/* Connector line — now #2563EB */}
                {i < steps.length - 1 && (
                  <div
                    style={{
                      marginLeft: 25,
                      width: 2,
                      height: 40,
                      backgroundColor: "#2563EB",
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
