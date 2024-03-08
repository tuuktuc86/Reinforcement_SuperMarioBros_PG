# Reinforcement_SuperMarioBros_PG
Solve SuperMarioBros using PG Method

## 개요
본 프로젝트는 지능제어 수업에서 진행한 super mario bros 게임을 강화학습 알고리즘으로 사용하여 해결하는 과정을 담고 있습니다. 강화학습 알고리즘 중 Policy Gradient 알고리즘을 구현하여 게임을 해결하였습니다.

## 프로젝트 설명
Super Mario Bros는 강화학습 알고리즘의 테스트 환경을 많이 사용되었습니다. 기본적인 게임 방법에 대한 내용은 다음 글을 참고 바랍니다.<br><br>
[pytorch 슈퍼마리오](https://tutorials.pytorch.kr/intermediate/mario_rl_tutorial.html)
<br><br>
프로젝트의 근간으로 삼은 코드는 아래 깃허브 주소와 같으며 Super Mario Bros 게임을 DQN알고리즘으로 구현한 코드입니다. 해당 프로젝트를 기반으로 Policy Gradient를 적용하였습니다.<br><br>
[github.ocm/yfeng997](https://github.com/yfeng997/MadMario)

## 모델 구조
<figure>
  <img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/play_video/model.png">
</figure>

## 그래프
각 데이터는 20step의 movig average를 적용한 데이터입니다.
|Length|Loss|Reward|
|---|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/checkpoints/2023-12-14T23-31-58/length_plot.jpg">|<img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/checkpoints/2023-12-14T23-31-58/loss_plot.jpg"> |<img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/checkpoints/2023-12-14T23-31-58/reward_plot.jpg">|


## 동영상

stage1-1에서 학습사여 stage1-2에서는 학습 수준이 부족합니다.<br>

|stage1-1|stage1-2|
|---|---|
|<img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/play_video/stage1_paly-3.gif" >|<img src="https://github.com/tuuktuc86/Reinforcement_SuperMarioBros_PG/blob/main/play_video/stage2_paly.gif">|
