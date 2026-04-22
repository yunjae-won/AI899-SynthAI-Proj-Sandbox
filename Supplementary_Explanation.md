[Skip to content](https://chiseled-headstand-8fc.notion.site/34741406b28d803397e9eb4fa2bc7a47#main)

# 중간발표

agent\_project/
├── README.md ← 프로젝트 개요 + 산출물 매핑
├── situation.md ← 기존 원형 이벤트 2개
├── schema/
│ ├── event\_schema.json ← 이벤트 JSON 스키마
│ ├── persona\_schema.json ← 페르소나 JSON 스키마
│ └── schema.md ← 설계 원칙·필드 해설
├── events/
│ ├── event\_01\_deadline\_compression.md + .json
│ └── event\_02\_campus\_routine.md + .json
├── personas/
│ ├── persona\_01\_balanced\_sophomore.md + .json
│ ├── persona\_02\_tired\_achiever.md + .json
│ └── persona\_03\_social\_explorer.md + .json
└── analysis/
└── analysis.md ← 3축 평가 기준 + 커버리지 매트릭스

​

schema/ → 이 부분은 sandbox 제작 시 변형해서 맞게 사용해도 될 것 같습니다!

#### 평가

각 이벤트 실행이 끝나면 아래와 같이 점수를 매김.

각 이벤트의 evaluation 필드에 관찰 가능한 지표가 미리 정의돼 있음 (event~.json 참고)

| 축 | 무엇을 보는가 | 측정 방법 |
| --- | --- | --- |
| Hard vs Soft | 되돌릴 수 없는 손실을 방어했는가 | hard\_loss\_conditions<br>리스트 중 몇 개 발생했는지 카운트 |
| Context Switching | 한 결정이 다른 도메인에 미치는 영향을 고려했는가 | 추론 trace에서 교차 도메인 언급 빈도 |
| Efficiency | 극단 선택(올나이트, 전면 포기) 대신 제3의 경로를 찾았는가 | efficiency\_indicators<br>행동의 등장 여부 |

→ events 폴더 내 json 파일 참고

채점 입력 (예시)

에이전트 실행 trajectory:

(timestamp, state\_snapshot, action, observation)

튜플 시퀀스. 이걸 후처리로 스캔하면서 각 축의 지표를 매칭.

Pass/Fail 기준

Hard loss 가 1개라도 발생 → 해당 이벤트 실패

Hard loss 없음 + context/efficiency 지표 절반 이상 충족 → 통과

모두 충족 \+ good\_response\_exemplar 와 유사 → 우수

이벤트×페르소나 매트릭스 평가

단일 이벤트가 아니라 페르소나 교체 실험으로 에이전트 능력을 측정: 같은 이벤트를 3개 페르소나로 각각 실행

위험군 페르소나(예: Event 01 × Tired Achiever)에서 에이전트가 페르소나 성향을 거스르며 hard loss를 방어했는지가 핵심 지표

naive한 에이전트는 페르소나 가치만 그대로 따라가 의사결정에 실패함

#### 페르소나

이벤트에 진입하는 에이전트의 초기 프로필. 같은 이벤트라도 누가 풀면 올바른 대응이 달라짐.

| 페르소나 | 역할 | 핵심 약점 | 위험 이벤트 |
| --- | --- | --- | --- |
| Balanced Sophomore | baseline | 없음 — 비교용 | — |
| Tired Achiever | 학업 0.55 극단 + 회피형 | Hard/Soft 구분 실패, 재협상 기피 | Event 01 (마감 압축) |
| Social Explorer | 관계 0.40 + 수용형 | 자기 계획 과소평가, 경계 설정 실패 | Event 02 (공강) |

정반대 약점 을 가진 두 극단 \+ 중간 baseline

에이전트가 페르소나 성향을 거슬러 상황에 맞게 유연한 판단을 하는지가 핵심

페르소나의 values만 에이전트가 그대로 따라가면 hard loss 발생

스키마 핵심 필드

baseline\_state

— energy/sleep\_debt/stress (평소 상태)

value\_priorities

— 학업/관계/건강/성장/여가 가중치 (합 ≈ 1)

communication\_style

— conflict\_style (avoidant/accommodating/compromising...)

relationship\_network

— 역할별 trust/closeness (페르소나와 연결된 주변 인물과의 관계)

decision\_tendencies

— planning\_horizon, risk\_tolerance, sleep\_discipline (페르소나가 의사결정을 내리는 스타일을 수치화함)

| 필드 | 값 | 행동에 미치는 영향 |
| --- | --- | --- |
| planning\_horizon | short / medium / long | 얼마나 먼 미래까지 고려하는가 |
| risk\_tolerance | 0~10 | 불확실한 선택을 얼마나 감수하는가 |
| procrastination\_bias | 0~10 | 미루는 성향. 높을수록 마감 임박 착수 |
| sleep\_discipline | 0~10 | 수면을 지키는 규율. 낮을수록 올나이트 기본값 |

ex) Event 01 에서 Tired Achiever 가 실패하는 이유

planning\_horizon: short

→ 다음날 퀴즈 페널티 과소평가

sleep\_discipline: 2

→ 올나이트를 자연스러운 옵션으로 선택

risk\_tolerance: 7

→ 조금만 더 버티면 된다라는 리스크 감수

→ 세 값이 합쳐져 올나이트 \+ 퀴즈 실패의 경로가 기본값이 됨