# Reinforcement_SuperMarioBros_PG
Solve SuperMarioBros using PG Method

## 개요
본 프로젝트는 지능제어 수업에서 진행한 super mario bros 게임을 강화학습 알고리즘으로 사용하여 해결하는 과정을 담고 있습니다. 다양한 알고리즘 중 Policy Gradient 알고리즘을 구현하여 게임을 해결하였습니다.

## 기반 프로젝트 및 참고 코드
Super Mario Bros는 강화학습 알고리즘의 테스트 환경을 많이 사용되었습니다. 기본적인 게임 방법에 대한 내용은 다음 글을 참고 바랍니다.
https://tutorials.pytorch.kr/intermediate/mario_rl_tutorial.html

다양한 github 코드가 있지만 이번 프로젝트에서 근간으로 삼은 코드는 

## PG 알고리즘이란?


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
