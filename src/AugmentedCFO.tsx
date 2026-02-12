import React from "react";
import { AbsoluteFill, Series } from "remotion";
import { TransitionSeries } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { linearTiming } from "@remotion/transitions";
import { IntroScene } from "./scenes/IntroScene";
import { ProblemScene } from "./scenes/ProblemScene";
import { SolutionScene } from "./scenes/SolutionScene";
import { ResultsScene } from "./scenes/ResultsScene";
import { CTAScene } from "./scenes/CTAScene";

export const AugmentedCFO: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#0a0a1a" }}>
      <TransitionSeries>
        {/* Scene 1: Intro / Brand - 4 seconds */}
        <TransitionSeries.Sequence durationInFrames={120}>
          <IntroScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          timing={linearTiming({ durationInFrames: 15 })}
          presentation={fade()}
        />

        {/* Scene 2: The Problem - 3 seconds */}
        <TransitionSeries.Sequence durationInFrames={90}>
          <ProblemScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          timing={linearTiming({ durationInFrames: 15 })}
          presentation={fade()}
        />

        {/* Scene 3: The Solution - 3 seconds */}
        <TransitionSeries.Sequence durationInFrames={90}>
          <SolutionScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          timing={linearTiming({ durationInFrames: 15 })}
          presentation={fade()}
        />

        {/* Scene 4: Results / Stats - 3 seconds */}
        <TransitionSeries.Sequence durationInFrames={90}>
          <ResultsScene />
        </TransitionSeries.Sequence>
        <TransitionSeries.Transition
          timing={linearTiming({ durationInFrames: 15 })}
          presentation={fade()}
        />

        {/* Scene 5: CTA - 3 seconds */}
        <TransitionSeries.Sequence durationInFrames={90}>
          <CTAScene />
        </TransitionSeries.Sequence>
      </TransitionSeries>
    </AbsoluteFill>
  );
};
