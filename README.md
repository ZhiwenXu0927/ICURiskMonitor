# ICURiskPredictor

Many scoring systems have been developed and machine learning algorithms have been built to analyze Electronic Health Record (EHR)
to monitor mortality risk for Intensive Care Unit(ICU) patients. However, there are two limitations in many current methods that make prediction tasks difficult: irregular timing of medical events and lack of explainablilty of black-box deep learning models. It is also difficult tosolve the two challenges at the same time as capturing this irregularity in timing often requires complicated models which do not have inherent explanations. This project builds a framework that incorporates neural networks that can represent and learn from irregular timing and and a post-hoc SHAP explainer that calculates feature importance given a data point. The SHAP values can help to identify key features, sanity check if the decision rules learned by the model align with clinical knowledge, and provide post-hoc analytics, e.g. patient clustering, for stakeholders for further evaluation. This framework can be further extended to more explainers and be more generic and applicable to more models.

Model training can be executed as illustrated in src/training.ipynb

SHAP explainer can be executed as illustrated in explainer_gru.ipynb/explanation_insights.ipynb
