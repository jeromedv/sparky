import React from "react";
import { AbsoluteFill, Sequence } from "remotion";
import { SceneTransition } from "./SceneTransition";
import { Scene1_LogoIntro } from "./scenes/Scene1_LogoIntro";
import { Scene2_PainQ1 } from "./scenes/Scene2_PainQ1";
import { Scene3_PainQ2 } from "./scenes/Scene3_PainQ2";
import { Scene4_PainQ3 } from "./scenes/Scene4_PainQ3";
import { Scene5_PainQ4 } from "./scenes/Scene5_PainQ4";
import { Scene6_Diagnosis } from "./scenes/Scene6_Diagnosis";
import { Scene7_SolutionReveal } from "./scenes/Scene7_SolutionReveal";
import { Scene8_BeforeAfter } from "./scenes/Scene8_BeforeAfter";
import { Scene9_HighValueWork } from "./scenes/Scene9_HighValueWork";
import { Scene10_HowItWorks } from "./scenes/Scene10_HowItWorks";
import { Scene11_Services } from "./scenes/Scene11_Services";
import { Scene12_CTAFinal } from "./scenes/Scene12_CTAFinal";

/**
 * ~84.5-second video (2535 frames @ 30fps) — Dark Premium
 *
 *  #  Scene                  Time        Frames       Duration
 *  1  Logo Intro             0:00–0:04   0–119        120 fr
 *  2  Pain Question 1        0:04–0:08   120–239      120 fr
 *  3  Pain Question 2        0:08–0:12   240–359      120 fr
 *  4  Pain Question 3        0:12–0:16   360–479      120 fr
 *  5  Pain Question 4        0:16–0:20   480–599      120 fr
 *  6  Diagnosis              0:20–0:25   600–749      150 fr
 *  7  Solution Reveal        0:25–0:31   750–929      180 fr
 *  8  Before/After Metrics   0:31–0:45   930–1364     435 fr
 *  9  High-Value Work        0:45–0:51   1365–1544    180 fr
 * 10  How It Works           0:51–1:00   1545–1814    270 fr
 * 11  Services               1:00–1:06   1815–1994    180 fr
 * 12  CTA Final              1:06–1:24   1995–2534    540 fr
 */
export const AugmentedCFO: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#0A0F1E" }}>
      <Sequence from={0} durationInFrames={120}>
        <SceneTransition durationInFrames={120}>
          <Scene1_LogoIntro />
        </SceneTransition>
      </Sequence>

      <Sequence from={120} durationInFrames={120}>
        <SceneTransition durationInFrames={120}>
          <Scene2_PainQ1 />
        </SceneTransition>
      </Sequence>

      <Sequence from={240} durationInFrames={120}>
        <SceneTransition durationInFrames={120}>
          <Scene3_PainQ2 />
        </SceneTransition>
      </Sequence>

      <Sequence from={360} durationInFrames={120}>
        <SceneTransition durationInFrames={120}>
          <Scene4_PainQ3 />
        </SceneTransition>
      </Sequence>

      <Sequence from={480} durationInFrames={120}>
        <SceneTransition durationInFrames={120}>
          <Scene5_PainQ4 />
        </SceneTransition>
      </Sequence>

      <Sequence from={600} durationInFrames={150}>
        <SceneTransition durationInFrames={150}>
          <Scene6_Diagnosis />
        </SceneTransition>
      </Sequence>

      <Sequence from={750} durationInFrames={180}>
        <SceneTransition durationInFrames={180}>
          <Scene7_SolutionReveal />
        </SceneTransition>
      </Sequence>

      <Sequence from={930} durationInFrames={435}>
        <SceneTransition durationInFrames={435}>
          <Scene8_BeforeAfter />
        </SceneTransition>
      </Sequence>

      <Sequence from={1365} durationInFrames={180}>
        <SceneTransition durationInFrames={180}>
          <Scene9_HighValueWork />
        </SceneTransition>
      </Sequence>

      <Sequence from={1545} durationInFrames={270}>
        <SceneTransition durationInFrames={270}>
          <Scene10_HowItWorks />
        </SceneTransition>
      </Sequence>

      <Sequence from={1815} durationInFrames={180}>
        <SceneTransition durationInFrames={180}>
          <Scene11_Services />
        </SceneTransition>
      </Sequence>

      <Sequence from={1995} durationInFrames={540}>
        <SceneTransition durationInFrames={540}>
          <Scene12_CTAFinal />
        </SceneTransition>
      </Sequence>
    </AbsoluteFill>
  );
};
