import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { Scene1_LogoIntro } from "./scenes/Scene1_LogoIntro";
import { Scene2_PainQuestions } from "./scenes/Scene2_PainQuestions";
import { Scene3_Diagnosis } from "./scenes/Scene3_Diagnosis";
import { Scene4_SolutionReveal } from "./scenes/Scene4_SolutionReveal";
import { Scene5_BeforeAfter } from "./scenes/Scene5_BeforeAfter";
import { Scene6_HighValueWork } from "./scenes/Scene6_HighValueWork";
import { Scene7_HowItWorks } from "./scenes/Scene7_HowItWorks";
import { Scene8_Services } from "./scenes/Scene8_Services";
import { Scene9_FinalCTA } from "./scenes/Scene9_FinalCTA";

/**
 * 75-second video (2250 frames @ 30fps)
 *
 * Scene 1: Logo Intro        0:00–0:02   frames    0–59    (60 frames)
 * Scene 2: Pain Questions     0:02–0:13   frames   60–389  (330 frames)
 * Scene 3: Diagnosis          0:13–0:18   frames  390–539  (150 frames)
 * Scene 4: Solution Reveal    0:18–0:23   frames  540–689  (150 frames)
 * Scene 5: Before/After       0:23–0:34   frames  690–1019 (330 frames)
 * Scene 6: High-Value Work    0:34–0:40   frames 1020–1199 (180 frames)
 * Scene 7: How It Works       0:40–0:50   frames 1200–1499 (300 frames)
 * Scene 8: Services Overview  0:50–0:56   frames 1500–1679 (180 frames)
 * Scene 9: Final CTA          0:56–1:15   frames 1680–2249 (570 frames)
 */
export const AugmentedCFO: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#FFFFFF" }}>
      <Sequence from={0} durationInFrames={60}>
        <Scene1_LogoIntro />
      </Sequence>

      <Sequence from={60} durationInFrames={330}>
        <Scene2_PainQuestions />
      </Sequence>

      <Sequence from={390} durationInFrames={150}>
        <Scene3_Diagnosis />
      </Sequence>

      <Sequence from={540} durationInFrames={150}>
        <Scene4_SolutionReveal />
      </Sequence>

      <Sequence from={690} durationInFrames={330}>
        <Scene5_BeforeAfter />
      </Sequence>

      <Sequence from={1020} durationInFrames={180}>
        <Scene6_HighValueWork />
      </Sequence>

      <Sequence from={1200} durationInFrames={300}>
        <Scene7_HowItWorks />
      </Sequence>

      <Sequence from={1500} durationInFrames={180}>
        <Scene8_Services />
      </Sequence>

      <Sequence from={1680} durationInFrames={570}>
        <Scene9_FinalCTA />
      </Sequence>
    </AbsoluteFill>
  );
};
