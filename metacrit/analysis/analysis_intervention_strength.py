"""Per-instance intervention strength for the camera-ready (Appendix B.7, Table 11).
d_i = 1 - cos(R_i, F_i); m_i = 1 - cos(R_i, V_i); c_i = 1 - cos(R_i, C_i).
Source: semantic_similarity_results.csv (text-embedding-3-small, same items as Fig. 3)."""
import pandas as pd, numpy as np
rng = np.random.default_rng(42)
rank = lambda x: pd.Series(x).rank().values
sp = lambda a, b: np.corrcoef(rank(a), rank(b))[0, 1]
def ci(a, b, B=5000):
    n = len(a); r = [sp(a[i], b[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(r, [2.5, 97.5])
s = pd.read_csv('results/semantic_similarity_results.csv')
for ds in ['TruthfulQA', 'CREAK', 'MMLU', 'BBQ', 'CrowS-Pairs']:
    g = s[s.dataset == ds]
    d = 1 - g.sim_background_analysis.values
    m = 1 - g.sim_background_validity.values
    c = 1 - g.sim_background_critique.values
    q = g.sim_to_answer_background.values
    print(ds, len(g), 'd p10-p90 %.3f-%.3f' % tuple(np.percentile(d, [10, 90])),
          'rho(m,d) %.2f %s' % (sp(m, d), ci(m, d).round(2)), 'rho(c,d) %.2f %s' % (sp(c, d), ci(c, d).round(2)),
          ('rho(q,d) %.2f %s' % (sp(q, d), ci(q, d).round(2))) if ds == 'TruthfulQA' else '')
