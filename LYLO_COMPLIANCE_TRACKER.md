# Lylo compliance and website tracker

Last updated: 17 September 2026

This file is the running list for the regulatory, privacy and website points raised while Lylo is being developed. "Done" means the website wording or document has been added. It does **not** mean Lylo has completed every operational compliance requirement behind that wording.

## Website and messaging

| Point | Status | What has been addressed | What remains |
|---|---|---|---|
| Scottish regulator/guidance should lead instead of SRA | Done | Homepage leads with the Law Society of Scotland Guide to Generative AI. | Keep guidance links reviewed as the site evolves. |
| Keep Aug 2026 SRA warning visible too | Done | Added a second tag: England & Wales · 17 Aug 2026 · SRA Misuse of AI warning. | Keep clearly labelled as England & Wales. |
| Keep heading “AI use in law is changing. Is your firm ready?” | Done | Restored on homepage. | — |
| Replace “legal judgement” wording | Done | “Lylo supports your informed decisions.” with “your” underlined. | — |
| Change UK GDPR “sits alongside” to “works alongside” | Done | Updated. | — |
| Change “built with duties…” wording | Done | “Lylo is designed with these responsibilities in mind from the beginning.” | — |
| Private AI rhetorical question | Done | Homepage asks whether the firm could use AI benefits without sending sensitive client data to a public AI service. | Keep wording aligned with actual deployment architecture. |
| Capitalise Engineering / Workflow role labels | Done | Product & Engineering; Legal Research & Workflow. | — |
| Co-founder wording | Done | Homepage link changed to “Meet the co-founders of Lylo”. | — |
| Research wording should avoid repeating “build/shape” | Done | Changed to “develop” and “guide Lylo’s development”. | — |
| Research note capitalisation | Done | “Used for Lylo product research.” | — |
| Jess instead of Jessica in About profile sentence | Done | About page says “Jess brings legal education…” | — |

## Data-protection framework on website

| Point | Status | What has been addressed | What remains |
|---|---|---|---|
| UK GDPR | Done on website | Referenced in homepage and privacy notice. | Operational compliance still requires documented controls. |
| Data Protection Act 2018 | Done on website | Added to homepage framework wording and privacy notice. | Keep policies/processes aligned. |
| Data (Use and Access) Act 2025 | Done on website | Added as part of the current UK framework; not presented as a blanket justification for product research. | Review specific effects on Lylo processing as workflows develop. |
| PECR 2003 | Done on website | Mentioned where relevant to electronic marketing, cookies and communications. | Review outreach/cookie/communications activity against PECR. |
| Clear privacy notice | Done — first version | Added `/static/privacy.html` and linked from homepage and questionnaire. | Replace/expand details as provider inventory, retention and controller/processor roles are finalised. |
| Individual rights / SARs | Done on website | Privacy notice explains access, correction, deletion requests and SARs, with appropriate qualification. | Create an internal request-handling procedure and response log. |
| Deletion requests | Done on website | Privacy notice explains deletion may be requested depending on circumstances. | Document exemptions, identity checks and response process. |

## Operational compliance work

| Point | Status | Current position | Next action |
|---|---|---|---|
| ICO fee / registration assessment | Not yet done | Website does not claim Lylo is registered. | Complete the ICO self-assessment now rather than waiting automatically for the first firm. Record the result. |
| Third-party / processor inventory | In progress | Privacy notice says Lylo uses providers for functions such as hosting, email and voice/telephony and that a full inventory is being documented. | List every current provider, purpose, data received, location, retention, contract/DPA and sub-processors. Include the phone/voice stack. |
| Data retention policy | Not yet done | Privacy notice says a formal written retention schedule is being completed. | Define retention periods for questionnaire data, prospect/outreach data, demo data, logs and any future pilot matter data. |
| Written data-protection policy | Not yet done | Privacy notice identifies this as outstanding. | Draft and approve an internal policy covering roles, lawful bases, minimisation, access, security, breaches, rights and retention. |
| Founder/staff data-protection training | Not yet done / do not claim yet | No public claim that Joshua or Jess has completed formal training. | Complete appropriate training, record dates/materials, then update site wording if useful. |
| Client-data handling training and confidentiality procedures | Not yet done | No claim that Lylo's model is “trained on client data”. | Train people who may access personal/client information and document handling/confidentiality rules. |
| Subject access request procedure | Not yet done | Website provides a contact route. | Create identity-check, search, review, response and logging procedure. |
| Data-deletion procedure | Not yet done | Website states the right in qualified form. | Create operational deletion/restriction workflow and record exceptions. |
| Data processing agreements with firms | Not yet done | Privacy notice states firm-specific terms are needed before live client-data pilots. | Determine controller/processor roles for each pilot and execute appropriate DPA terms. |
| DPIA / risk assessment for high-risk workflows | Not yet done | Not claimed as complete. | Assess before live sensitive legal-data use, particularly where new AI/voice processing could create high risk. |
| Security controls for live client-data pilot | Not yet done | Current site describes intended private architecture, not a certified production environment. | Document access control, encryption, backups, logging, incident response, patching and deployment responsibilities before live client data. |
| Breach/incident response procedure | Not yet done | Not currently documented publicly. | Create incident response and breach-assessment process including ICO/client notification decision points. |
| Records of processing activities | Not yet done | No formal ROPA recorded here. | Create processing inventory and determine whether formal Article 30 records are required; maintain regardless as good governance. |
| Cookie/analytics review | Not yet done | PECR is mentioned in privacy notice. | Identify any cookies/analytics actually used and add consent/banner controls if required. |
| Marketing/outreach PECR review | Not yet done | No blanket claim of compliance. | Review cold email/phone/DM workflows by recipient type and channel before scaling outreach. |

## Product research

| Point | Status | Current position | Next action |
|---|---|---|---|
| Questionnaire is for product research | Done | Questionnaire and homepage say this clearly. | Keep purpose consistent with actual use of responses. |
| DUAA 2025 and research | Corrected | Site does not claim DUAA automatically authorises commercial product research. It is listed as part of the wider legal framework. | Confirm lawful basis and transparency for each research dataset. |
| Do not collect confidential client information in questionnaire | Done | Clear warning appears next to submission and in privacy notice. | Keep form design and internal handling aligned. |

## Rule for future updates

When a new compliance or privacy point is raised:
1. Add it to this tracker.
2. Separate **website wording** from **operational work**.
3. Do not mark operational work as complete merely because wording was added to the website.
4. Do not make public claims until the underlying control, training, registration, contract or process actually exists.
