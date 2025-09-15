# AdminOpt8

**AdminOpt8** est un moteur **TDSA** (Théorème de Décision Séquentielle Administrative) prêt à l’emploi.  
Il permet de prendre des décisions administratives binaires (**accepter/refuser**) en minimisant les coûts,  
en s’appuyant sur un **test séquentiel optimal** (SPRT généralisé).

👉 L’outil peut servir dans tout contexte de validation de dossiers, contrôles, audits, éligibilités.  
Il est conçu pour **réduire drastiquement le nombre de pièces justificatives collectées**,  
**baisser les coûts de contrôle** et **améliorer la traçabilité**.

---

## ✨ Fonctionnalités

- Moteur TDSA (test séquentiel optimal SPRT)
- API REST complète (FastAPI)
- Mini interface web HTML/JS intégrée (aucune dépendance front)
- Catalogue de "preuves" (evidences) avec coût, TPR, FPR
- Choix adaptatif de la prochaine preuve par **info/coût**
- Journalisation par session (trace détaillée, export JSON)
- Support de catalogue custom (POST `/api/catalog`)

---

## 🚀 Démarrage rapide

### Installation

```bash
git clone https://github.com/toncompte/adminopt8.git
cd adminopt8
pip install -r requirements.txt
