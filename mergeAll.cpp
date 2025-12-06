#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <numeric>   // for accumulate
#include <algorithm> // for max_element, min, max
#include <cmath>     // for sqrt, exp, llround
#include <ctime>     // for time
#include <cstdlib>   // for rand, srand
#include <tuple>     // for tuple

using namespace std;

const int HOURS = 24;

// ========================= Tariff (CPP) =========================
vector<double> tariff_CPP = {
    30,30,30,30,30,             // 0-4 base
    140,140,140,140,140,140,    // 5-10 critical peak
    30,32,36,44,56,60,          // 11-16 evening high
    52,46,40,36,32,30,30        // 17-23
};

// ========================= Appliance data =========================
// type: 0 = fixed, 1 = non-shiftable (limited window), 2 = shiftable
struct Appliance {
    string name;
    int duration;   // contiguous hours
    double power;   // kW
    int type;       // 0 = fixed, 1 = non-shiftable, 2 = shiftable
    int baseline;   // baseline start
    int window;     // half-window size
    Appliance(string n, int d, double p, int tp, int b, int w = 3)
        : name(n), duration(d), power(p), type(tp), baseline(b), window(w) {}
};

// Realistic household appliances
vector<Appliance> appliances = {
    {"Air Conditioner",      4,   2.0,   2, 18, 3},
    {"Air Purifier",         6,   0.1,   2,  0, 3},
    {"Coffee Maker",         1,   0.4,   1,  7, 2},
    {"Computer",             4,   0.25,  1, 20, 3},
    {"Digital Clock",       24,   0.0025,0,  0, 0},
    {"Dishwasher",           3,   1.33,  2, 21, 4},
    {"Hair Dryer",           1,   2.0,   1,  7, 2},
    {"Electric Iron",        1,   2.0,   1,  7, 2},
    {"EV Charger",           8,   4.0,   2, 22, 6},
    {"Exhaust Fan",          2,   0.1,   2,  8, 3},
    {"Fan",                  8,   0.1,   2, 20, 4},
    {"Food Blender",         1,   0.4,   1,  8, 2},
    {"Induction Cooker",     1,   2.0,   1, 19, 2},
    {"LED Lights",           5,   0.08,  1, 19, 4},
    {"Microwave",            1,   1.5,   1, 12, 2},
    {"Night Light",          2,   0.05,  0, 22, 0},
    {"Refrigerator",         6,   0.3,   0,  0, 0},
    {"Room Heater",          3,   2.0,   2,  6, 3},
    {"Router WiFi",         24,   0.025, 0,  0, 0},
    {"Shaver",               1,   0.05,  1,  6, 2},
    {"Television",           3,   0.2,   1, 20, 3},
    {"Vacuum Cleaner",       1,   1.0,   2, 10, 8},
    {"Washing Machine",      2,   0.6,   2,  9, 8},
    {"Water Heater",         1,   2.5,   1,  6, 2},
    {"Water Pump",           1,   1.5,   1,  5, 2}
};

// ========================= Bounds per appliance =========================
struct Bounds { int lo, hi; }; // inclusive

Bounds get_bounds_for_appliance(int idx) {
    const Appliance &ap = appliances[idx];
    int dur = ap.duration;
    int lo, hi;

    if (ap.type == 0) {
        // Fixed load: always at baseline
        int s = min(ap.baseline, HOURS - dur);
        if (s < 0) s = 0;
        lo = hi = s;
    } else if (ap.type == 1) {
        // Non-shiftable: limited window around baseline
        lo = max(0, ap.baseline - ap.window);
        hi = min(HOURS - dur, ap.baseline + ap.window);
    } else {
        // Shiftable: can be anywhere
        lo = 0;
        hi = HOURS - dur;
    }
    if (hi < lo) hi = lo; // safety
    return {lo, hi};
}

// ========================= Basic helpers =========================
vector<int> default_unscheduled_starts() {
    vector<int> s;
    s.reserve(appliances.size());
    for (auto &ap : appliances) s.push_back(ap.baseline);
    return s;
}

double cost_from_starts(const vector<int>& start, const vector<double>& tariff) {
    double cost = 0.0;
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        // Safety check to prevent out of bounds access
        for (int h = s; h < s + d && h < HOURS; ++h) {
            cost += p * tariff[h];
        }
    }
    return cost;
}

vector<double> load_profile_from_starts(const vector<int>& start) {
    vector<double> load(HOURS, 0.0);
    for (int a = 0; a < (int)appliances.size(); a++) {
        int s = start[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = s; h < s + d && h < HOURS; ++h) {
            load[h] += p;
        }
    }
    return load;
}

// PAR = peak / average
double compute_PAR(const vector<int>& start) {
    vector<double> load = load_profile_from_starts(start);
    double peak = *max_element(load.begin(), load.end());
    double sum = accumulate(load.begin(), load.end(), 0.0);
    double avg = sum / HOURS;
    return peak / (avg + 1e-12);
}

void print_schedule_starts(const vector<int>& start) {
    for (int a = 0; a < (int)appliances.size(); a++) {
        cout << setw(2) << a << " : " << setw(20) << appliances[a].name
             << " start=" << setw(2) << start[a]
             << " dur=" << setw(2) << appliances[a].duration
             << " type=" << (appliances[a].type==0 ? "fixed" :
                             appliances[a].type==1 ? "non-shift" : "shift")
             << "\n";
    }
}

// ========================= Load shifting helpers =========================
double mean(const vector<double>& v){
    double s=0; for(double x:v) s+=x; return s / (double)v.size();
}
double stdev(const vector<double>& v){
    double m=mean(v); double s=0; for(double x:v) s+=(x-m)*(x-m); return sqrt(s / (double)v.size());
}
vector<double> normalize_vector(const vector<double>& v){
    double mn = *min_element(v.begin(), v.end());
    double mx = *max_element(v.begin(), v.end());
    vector<double> out(v.size());
    double denom = (mx - mn) + 1e-12;
    for (size_t i=0;i<v.size();++i) out[i] = (v[i] - mn) / denom;
    return out;
}

struct LoadLimits {
    vector<double> PL_unsch, PL_N;
    double Plo, PLoff, PLon;
};

vector<double> compute_hourly_load(const vector<int>& starts) {
    vector<double> L(HOURS, 0.0);
    for (size_t a = 0; a < appliances.size(); ++a) {
        int s = starts[a];
        int d = appliances[a].duration;
        double p = appliances[a].power;
        for (int h = 0; h < d && s + h < HOURS; ++h) L[s + h] += p;
    }
    return L;
}

LoadLimits compute_load_limits_from_unscheduled(const vector<double>& PL_unsch, double eta=2.0) {
    LoadLimits L;
    L.PL_unsch = PL_unsch;
    L.PL_N = normalize_vector(PL_unsch);
    L.Plo = mean(PL_unsch);
    double std_uns = stdev(PL_unsch);
    double sumPLN = 0.0; for (double x : L.PL_N) sumPLN += x;
    L.PLoff = sumPLN - std_uns;
    double min_uns = *min_element(PL_unsch.begin(), PL_unsch.end());
    L.PLon = std_uns - eta * min_uns;

    // Safeguards
    if (L.PLoff > L.PLon) swap(L.PLoff, L.PLon);
    if (L.PLoff < 0) L.PLoff = 0;
    if (L.PLoff > L.Plo) L.PLoff = L.Plo;
    if (L.PLon < L.Plo) L.PLon = L.Plo;
    return L;
}

vector<double> normalize_price(const vector<double>& tariff) {
    double mn = *min_element(tariff.begin(), tariff.end());
    double mx = *max_element(tariff.begin(), tariff.end());
    vector<double> p(HOURS);
    double denom = (mx - mn) + 1e-12;
    for (int t=0;t<HOURS;++t) p[t] = (tariff[t] - mn) / denom;
    return p;
}

vector<double> compute_PL_obj_from_price(const vector<double>& tariff,
                                         const LoadLimits& L,
                                         double K = 1.0) {
    vector<double> pnorm = normalize_price(tariff);
    vector<double> PLobj(HOURS, L.Plo);
    double mean_p = mean(pnorm);
    double span = (L.PLon - L.PLoff);
    double lo_bound = min(L.PLoff, L.PLon);
    double hi_bound = max(L.PLoff, L.PLon);
    for (int t=0;t<HOURS;++t) {
        double delta = (mean_p - pnorm[t]) * span * K;
        double val = L.Plo + delta;
        if (val < lo_bound) val = lo_bound;
        if (val > hi_bound) val = hi_bound;
        PLobj[t] = val;
    }
    return PLobj;
}

// F_loadshift (to be minimized)
double load_shifting_fitness(const vector<int>& starts,
                             const LoadLimits& limits,
                             const vector<double>& PLobj,
                             const vector<int>& off_peak_hours,
                             const vector<int>& on_peak_hours,
                             double penalty_weight = 1000.0) {
    vector<double> sched = compute_hourly_load(starts);
    double sse = 0.0;
    for (int t = 0; t < HOURS; ++t) {
        double diff = sched[t] - PLobj[t];
        sse += diff * diff;
    }

    double pen = 0.0;
    vector<char> is_off(HOURS,0), is_on(HOURS,0);
    for (int h : off_peak_hours) if (h>=0 && h<HOURS) is_off[h] = 1;
    for (int h : on_peak_hours)  if (h>=0 && h<HOURS) is_on[h]  = 1;
    for (int t = 0; t < HOURS; ++t) {
        if (is_off[t]) {
            if (sched[t] < limits.PLoff) {
                double v = limits.PLoff - sched[t];
                pen += v * v;
            }
        }
        if (is_on[t]) {
            if (sched[t] > limits.PLon)  {
                double v = sched[t] - limits.PLon;
                pen += v * v;
            }
        }
    }
    return sse + penalty_weight * pen;
}

// ========================= GA & AOA common stuff =========================
using Chromosome = vector<vector<int>>; // per-appliance 24-length 0/1 schedule

vector<int> schedule_from_start(int start, int dur) {
    vector<int> v(HOURS, 0);
    for (int i = start; i < start + dur && i < HOURS; ++i) v[i] = 1;
    return v;
}

// Create random chromosome respecting bounds
Chromosome ga_createChromosome() {
    Chromosome chromosome;
    chromosome.reserve(appliances.size());
    for (int i = 0; i < (int)appliances.size(); ++i) {
        const Appliance &ap = appliances[i];
        Bounds b = get_bounds_for_appliance(i);
        int dur = ap.duration;
        int range = b.hi - b.lo + 1;
        if (range <= 0) range = 1; // Safeguard
        int start = b.lo + (rand() % range);
        chromosome.push_back(schedule_from_start(start, dur));
    }
    return chromosome;
}

vector<int> ga_chromosome_to_starts(const Chromosome& ch) {
    vector<int> starts(ch.size(), 0);
    for (int a = 0; a < (int)ch.size(); ++a) {
        int first = -1;
        for (int h = 0; h < HOURS; ++h) {
            if (ch[a][h] == 1) { first = h; break; }
        }
        if (first < 0) {
            Bounds b = get_bounds_for_appliance(a);
            first = b.lo;
        }
        starts[a] = first;
    }
    return starts;
}

Chromosome ga_mutate(Chromosome ch) {
    const double MUTATION_RATE = 0.25;
    if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
        int idx = rand() % (int)appliances.size();
        const Appliance &ap = appliances[idx];
        if (ap.type == 0) return ch; // don't move fixed loads
        Bounds b = get_bounds_for_appliance(idx);
        int dur = ap.duration;
        int range = b.hi - b.lo + 1;
        if (range <= 0) range = 1;
        int start = b.lo + (rand() % range);
        ch[idx] = schedule_from_start(start, dur);
    }
    return ch;
}

pair<Chromosome, Chromosome> ga_crossover_pair(const Chromosome &p1, const Chromosome &p2) {
    const double CROSSOVER_RATE = 0.85;
    if (((double) rand() / RAND_MAX) < CROSSOVER_RATE) {
        int point = rand() % (int)appliances.size();
        Chromosome c1, c2;
        c1.reserve(appliances.size());
        c2.reserve(appliances.size());
        for (int i = 0; i < (int)appliances.size(); i++) {
            if (i < point) { c1.push_back(p1[i]); c2.push_back(p2[i]); }
            else           { c1.push_back(p2[i]); c2.push_back(p1[i]); }
        }
        return {c1, c2};
    }
    return {p1, p2};
}

// ========================= Multi-objective context =========================
struct MultiObjContext {
    vector<double> tariff;
    LoadLimits limits;
    vector<double> PLobj;
    vector<int> off_peak_hours;
    vector<int> on_peak_hours;
    double penalty_weight;

    double cost_unsched;
    double par_unsched;
    double ls_unsched;

    double alpha, beta, gamma; // weights: cost, PAR, loadshift
};

// Evaluate weighted objective for given starts
double evaluate_starts_weighted(const vector<int>& starts,
                                const MultiObjContext& ctx) {
    double cost = cost_from_starts(starts, ctx.tariff);
    double par  = compute_PAR(starts);
    double lsF  = load_shifting_fitness(starts, ctx.limits, ctx.PLobj,
                                       ctx.off_peak_hours, ctx.on_peak_hours,
                                       ctx.penalty_weight);

    double cn = cost / (ctx.cost_unsched + 1e-9);
    double pn = par  / (ctx.par_unsched  + 1e-9);
    double ln = lsF  / (ctx.ls_unsched   + 1e-9);

    return ctx.alpha * cn + ctx.beta * pn + ctx.gamma * ln;
}

double chromosome_weighted_fitness(const Chromosome &ch,
                                   const MultiObjContext& ctx) {
    vector<int> starts = ga_chromosome_to_starts(ch);
    return evaluate_starts_weighted(starts, ctx);
}

struct Metrics {
    double cost;
    double par;
    double lsF;
    double weighted;
};

Metrics compute_metrics(const vector<int>& starts,
                        const MultiObjContext& ctx) {
    Metrics m;
    m.cost = cost_from_starts(starts, ctx.tariff);
    m.par  = compute_PAR(starts);
    m.lsF  = load_shifting_fitness(starts, ctx.limits, ctx.PLobj,
                                  ctx.off_peak_hours, ctx.on_peak_hours,
                                  ctx.penalty_weight);
    double cn = m.cost / (ctx.cost_unsched + 1e-9);
    double pn = m.par  / (ctx.par_unsched  + 1e-9);
    double ln = m.lsF  / (ctx.ls_unsched   + 1e-9);
    m.weighted = ctx.alpha * cn + ctx.beta * pn + ctx.gamma * ln;
    return m;
}

// ========================= GA (multi-objective) =========================
pair<vector<int>, double> run_GA_multi(const MultiObjContext& ctx,
                                       bool verbose = true) {
    const int GA_POP_SIZE    = 16;
    const int GA_GENERATIONS = 50;

    vector<Chromosome> population;
    population.reserve(GA_POP_SIZE);
    for (int i = 0; i < GA_POP_SIZE; i++) population.push_back(ga_createChromosome());

    Chromosome best = population[0];
    double bestFit = chromosome_weighted_fitness(best, ctx);

    for (int gen = 0; gen < GA_GENERATIONS; gen++) {
        vector<double> fit(population.size());
        for (int i = 0; i < (int)population.size(); ++i)
            fit[i] = chromosome_weighted_fitness(population[i], ctx);

        // Roulette selection
        auto select_one = [&]() -> Chromosome {
            vector<double> weights(fit.size());
            double eps = 1e-9;
            for (int i = 0; i < (int)fit.size(); ++i)
                weights[i] = 1.0 / (fit[i] + eps);
            double sum = accumulate(weights.begin(), weights.end(), 0.0);
            double pick = ((double) rand() / RAND_MAX) * sum;
            double cur = 0.0;
            for (int i = 0; i < (int)population.size(); i++) {
                cur += weights[i];
                if (cur >= pick) return population[i];
            }
            return population.back();
        };

        vector<Chromosome> newpop;
        newpop.reserve(GA_POP_SIZE);
        // elitism
        for (int i = 0; i < (int)population.size(); ++i) {
            if (fit[i] < bestFit) { bestFit = fit[i]; best = population[i]; }
        }
        newpop.push_back(best);

        while ((int)newpop.size() < GA_POP_SIZE) {
            Chromosome p1 = select_one();
            Chromosome p2 = select_one();
            auto kids = ga_crossover_pair(p1, p2);
            Chromosome c1 = ga_mutate(kids.first);
            Chromosome c2 = ga_mutate(kids.second);
            newpop.push_back(c1);
            if ((int)newpop.size() < GA_POP_SIZE) newpop.push_back(c2);
        }

        population.swap(newpop);

        for (auto &ch : population) {
            double f = chromosome_weighted_fitness(ch, ctx);
            if (f < bestFit) { bestFit = f; best = ch; }
        }
        // if (verbose) cout << "GA Gen " << gen+1 << ": Best F = " << bestFit << "\n";
    }

    vector<int> bestStarts = ga_chromosome_to_starts(best);
    return {bestStarts, bestFit};
}

// ========================= AOA (multi-objective) =========================
inline double urand() { return (double)rand() / (double)RAND_MAX; }

void discretize_round_clamp(const vector<double>& x,
                            vector<int>& start,
                            const vector<Bounds>& B) {
    start.resize(x.size());
    for (int i = 0; i < (int)x.size(); ++i) {
        int v = (int)llround(x[i]);
        if (v < B[i].lo) v = B[i].lo;
        if (v > B[i].hi) v = B[i].hi;
        // enforce fixed loads exactly at baseline
        if (appliances[i].type == 0) v = appliances[i].baseline;
        start[i] = v;
    }
}

pair<vector<int>, double> run_AOA_multi(const MultiObjContext& ctx,
                                        const vector<int>& seedStart,
                                        bool seedProvided,
                                        bool verbose = true) {
    const int AOA_POP  = 28;
    const int AOA_ITER = 50;
    const double C1 = 2.0, C2 = 6.0, C3 = 2.0, C4 = 1.0, U = 0.9, L = 0.1;

    int D = (int)appliances.size();
    vector<Bounds> B(D);
    for (int d = 0; d < D; ++d) B[d] = get_bounds_for_appliance(d);

    vector<vector<double>> X(AOA_POP, vector<double>(D));
    vector<vector<double>> DEN(AOA_POP, vector<double>(D));
    vector<vector<double>> VOL(AOA_POP, vector<double>(D));
    vector<vector<double>> ACC(AOA_POP, vector<double>(D));

    auto rand_in_bounds = [&](int d){
        return B[d].lo + urand() * (B[d].hi - B[d].lo);
    };

    // init
    for (int i = 0; i < AOA_POP; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i][d]   = rand_in_bounds(d);
            DEN[i][d] = urand();
            VOL[i][d] = urand();
            ACC[i][d] = urand();
        }
    }
    // seed best individual with GA result
    if (seedProvided) {
        for (int d = 0; d < D; ++d) X[0][d] = (double)seedStart[d];
    }

    double bestFit = 1e100;
    vector<double> Xbest(D), DENbest(D), VOLbest(D), ACCbest(D);

    // evaluate initial best
    for (int i = 0; i < AOA_POP; ++i) {
        vector<int> s;
        discretize_round_clamp(X[i], s, B);
        double f = evaluate_starts_weighted(s, ctx);
        if (f < bestFit) {
            bestFit = f;
            Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i];
        }
    }

    vector<vector<double>> DENn(AOA_POP, vector<double>(D));
    vector<vector<double>> VOLn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCnorm(AOA_POP, vector<double>(D));

    for (int t = 1; t <= AOA_ITER; ++t) {
        // 1) dens/vol to best
        for (int i = 0; i < AOA_POP; ++i) {
            for (int d = 0; d < D; ++d) {
                DENn[i][d] = DEN[i][d] + urand() * (DENbest[d] - DEN[i][d]);
                VOLn[i][d] = VOL[i][d] + urand() * (VOLbest[d] - VOL[i][d]);
            }
        }

        // 2) transfer and decay
        double TF = exp((double(t) - AOA_ITER) / (double)AOA_ITER);
        double dens_decay = exp((double(AOA_ITER) - t) / (double)AOA_ITER) - (double)t / (double)AOA_ITER;

        // 3) acceleration
        for (int i = 0; i < AOA_POP; ++i) {
            if (TF <= 0.5) {
                int mr = rand() % AOA_POP;
                for (int d = 0; d < D; ++d) {
                    double num  = DEN[mr][d] + VOL[mr][d] * ACC[mr][d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            } else {
                for (int d = 0; d < D; ++d) {
                    double num  = DENbest[d] + VOLbest[d] * ACCbest[d];
                    double deno = DENn[i][d] + VOLn[i][d];
                    ACCn[i][d] = num / (deno + 1e-12);
                }
            }
        }

        // 4) normalize acc
        for (int d = 0; d < D; ++d) {
            double mn = 1e100, mx = -1e100;
            for (int i = 0; i < AOA_POP; ++i) {
                mn = min(mn, ACCn[i][d]);
                mx = max(mx, ACCn[i][d]);
            }
            double denom = (mx - mn) + 1e-12;
            for (int i = 0; i < AOA_POP; ++i) {
                double z = (ACCn[i][d] - mn) / denom;
                ACCnorm[i][d] = U * z + L;
            }
        }

        // 5) update positions
        for (int i = 0; i < AOA_POP; ++i) {
            if (TF <= 0.5) {
                int rr = rand() % AOA_POP;
                for (int d = 0; d < D; ++d) {
                    double step = C1 * urand() * ACCnorm[i][d] * dens_decay * (X[rr][d] - X[i][d]);
                    X[i][d] += step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            } else {
                double Tpar = C3 * TF;
                double P = 2.0 * urand() - C4;
                double F = (P <= 0.5) ? +1.0 : -1.0;
                for (int d = 0; d < D; ++d) {
                    double step = F * C2 * urand() * ACCnorm[i][d] * dens_decay
                                  * (Tpar * Xbest[d] - X[i][d]);
                    X[i][d] = Xbest[d] + step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            }
        }

        // 6) accept and update best
        DEN.swap(DENn); VOL.swap(VOLn); ACC.swap(ACCn);
        for (int i = 0; i < AOA_POP; ++i) {
            vector<int> s;
            discretize_round_clamp(X[i], s, B);
            double f = evaluate_starts_weighted(s, ctx);
            if (f < bestFit) {
                bestFit = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i];
            }
        }

        // if (verbose) cout << "AOA Iter " << t << ": Best F = " << bestFit << "\n";
    }

    vector<int> bestStart;
    discretize_round_clamp(Xbest, bestStart, B);
    return {bestStart, bestFit};
}

// ========================= MAIN =========================
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand((unsigned)time(0));

    cout << fixed << setprecision(4);

    // ---------- Baseline (unscheduled = baseline starts) ----------
    vector<int> baselineStarts = default_unscheduled_starts();
    vector<double> PL_unsch = compute_hourly_load(baselineStarts);
    LoadLimits limits = compute_load_limits_from_unscheduled(PL_unsch, 2.0);
    vector<double> PLobj = compute_PL_obj_from_price(tariff_CPP, limits, 1.0);

    vector<int> off_peak_hours;
    for (int h=0; h<=6; ++h) off_peak_hours.push_back(h);
    off_peak_hours.push_back(23);
    vector<int> on_peak_hours = {18,19,20,21};

    double penalty_weight = 5000.0;

    double cost_baseline = cost_from_starts(baselineStarts, tariff_CPP);
    double par_baseline  = compute_PAR(baselineStarts);
    double ls_baseline   = load_shifting_fitness(baselineStarts, limits, PLobj,
                                                 off_peak_hours, on_peak_hours,
                                                 penalty_weight);

    cout << "===== Baseline (Unscheduled) Metrics =====\n";
    cout << "Cost_baseline (Rs)  = " << cost_baseline << "\n";
    cout << "PAR_baseline        = " << par_baseline  << "\n";
    cout << "LoadShift_F_baseline= " << ls_baseline   << "\n\n";

    // ---------- Weight sets (alpha: cost, beta: PAR, gamma: load shifting) ----------
    vector<tuple<double,double,double>> weight_sets = {
        make_tuple(0.4, 0.35, 0.25),
        make_tuple(0.5, 0.30, 0.20),
        make_tuple(0.6, 0.25, 0.15)
    };

    struct Row {
        string method;
        Metrics m;
    };

    for (int w = 0; w < (int)weight_sets.size(); ++w) {
        double alpha = get<0>(weight_sets[w]);
        double beta  = get<1>(weight_sets[w]);
        double gamma = get<2>(weight_sets[w]);

        cout << "\n======================================================\n";
        cout << "Weight Set " << (w+1)
             << " : alpha=" << alpha
             << " (cost), beta=" << beta
             << " (PAR), gamma=" << gamma
             << " (load shifting)\n";
        cout << "======================================================\n";

        MultiObjContext ctx;
        ctx.tariff          = tariff_CPP;
        ctx.limits          = limits;
        ctx.PLobj           = PLobj;
        ctx.off_peak_hours  = off_peak_hours;
        ctx.on_peak_hours   = on_peak_hours;
        ctx.penalty_weight  = penalty_weight;
        ctx.cost_unsched    = cost_baseline;
        ctx.par_unsched     = par_baseline;
        ctx.ls_unsched      = ls_baseline;
        ctx.alpha           = alpha;
        ctx.beta            = beta;
        ctx.gamma           = gamma;

        // Baseline metrics in this context
        Metrics baseMetrics = compute_metrics(baselineStarts, ctx);

        // 1) GA phase
        // cout << "Running GA...\n";
        auto ga_res = run_GA_multi(ctx, true);
        vector<int> ga_best_starts = ga_res.first;
        Metrics gaMetrics = compute_metrics(ga_best_starts, ctx);

        // 2) AOA phase (seeded by GA)
        // cout << "Running AOA...\n";
        auto aoa_res = run_AOA_multi(ctx, ga_best_starts, true, true);
        vector<int> aoa_best_starts = aoa_res.first;
        Metrics aoaMetrics = compute_metrics(aoa_best_starts, ctx);

        // 3) Hybrid final (best of GA vs AOA by weighted objective)
        vector<int> final_starts;
        Metrics finalMetrics;
        if (aoaMetrics.weighted <= gaMetrics.weighted) {
            final_starts  = aoa_best_starts;
            finalMetrics  = aoaMetrics;
        } else {
            final_starts  = ga_best_starts;
            finalMetrics  = gaMetrics;
        }

        // ---------- Tabular output for this weight set ----------
        cout << "\nRESULT TABLE for Weight Set " << (w+1) << ":\n";
        cout << left
             << setw(12) << "Method"
             << setw(15) << "Cost (Rs)"
             << setw(12) << "PAR"
             << setw(18) << "LoadShift_F"
             << setw(18) << "Weighted F"
             << "\n";
        cout << string(75, '-') << "\n";

        auto print_row = [&](const string& name, const Metrics& m) {
            cout << left
                 << setw(12) << name
                 << setw(15) << m.cost
                 << setw(12) << m.par
                 << setw(18) << m.lsF
                 << setw(18) << m.weighted
                 << "\n";
        };

        print_row("Baseline", baseMetrics);
        print_row("GA",       gaMetrics);
        print_row("AOA",      aoaMetrics);
        print_row("Hybrid",   finalMetrics);

        cout << string(75, '-') << "\n";

        // (Optional) you can also print final schedule:
        // cout << "\nHybrid final schedule (starts) for this weight set:\n";
        // print_schedule_starts(final_starts);
        // cout << "\n";
    }

    cout << "\nDone. Press Enter to exit...";
    cin.ignore();
    cin.get();

    return 0;
}