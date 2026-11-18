# 缺失值處理會改變 PCA 看到的結構嗎？——nycflights13 分析

**作者**：林沛萱　**日期**：2026 年 11 月

## 互動成果
- 互動探索頁（Q1–Q7 問題導覽、hover、篩選、切分、框選）：https://peggy526.github.io/nycflights13-missing-pca/
- R 互動 notebook：https://peggy526.github.io/nycflights13-missing-pca/pca_explore.html

## 主要發現
- 延誤缺失是結構性缺失（航班取消或轉降）；氣壓缺失跟著當下天氣，較可能是 MAR。
- 直接刪除含缺失的航班，平均抵達延誤會被低估約 31%。
- 依機制設計的時間內插（策略 D）補值誤差最小。
- PC1（悶熱潮濕天氣與延誤）與 PC2（航程長度）在四種策略下都很穩定（相似度 ≥ 0.99）。

## 檔案說明
| 檔案 | 內容 |
| --- | --- |
| [report.pdf](report.pdf) | 完整報告 |
| [appendix_ai_baseline.pdf](appendix_ai_baseline.pdf) | AI 基準 EDA 報表 |
| [interactive_screenshots.pdf](interactive_screenshots.pdf) | 互動探索截圖與發現 |
| `main_analysis.Rmd` / `.html` | 資料合併、缺失診斷、四種策略、PCA |
| `pca_explore.Rmd` / `.html` | R 互動探索 |
| `dashboard_source/` | 互動探索頁的產生程式（Python，資料處理與 PCA 設定同 R 主程式） |

## 如何重現
1. 安裝 R 套件：`install.packages(c("tidyverse", "nycflights13", "naniar", "plotly", "crosstalk", "htmltools", "knitr", "lubridate"))`
2. knit `main_analysis.Rmd`，會產生 `pca_objects.RData`。
3. 再 knit `pca_explore.Rmd`。
