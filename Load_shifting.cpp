// loadshift_with_categories.cpp
// GA -> AOA hybrid for load-shifting with appliance categories (fixed / non-shiftable / shiftable)
// Compile: g++ loadshift_with_categories.cpp -O2 -o loadshift_with_categories
// Run: ./loadshift_with_categories

#include <bits/stdc++.h>
using namespace std;
const int HOURS = 24;

// ---------- Tariff (same as before) ----------
vector<double> tariff_default = {
    30,30,30,30,30,             // 0–4 base
    140,140,140,140,140,140,    // 5–10 critical peak
    30,32,36,44,56,60,          // 11–16 evening high
    52,46,40,36,32,30,30        // 17–23
};

// ---------- Appliance struct with category ----------
struct Appliance {
    string name;
    int duration;
    double power;
    int type;       // 0 = fixed, 1 = non-shiftable (limited window), 2 = shiftable (fully flexible)
    int baseline;   // baseline start used for fixed and as reference for non-shiftable
    int window;     // half-window size for non-shiftable (allowed shift is baseline +/- window)
    Appliance(string n,int d,double p,int tp,int b,int w=3)
      : name(n), duration(d), power(p), type(tp), baseline(b), window(w) {}
};

// ---------- Use same appliance list but with type and baseline start ---------
// I set baseline as an example; adjust as you prefer in default_unscheduled_starts() too.
vector<Appliance> appliances = {
    // name, duration (h), power (kW), type, baseline_start, window(for non-shiftable)
    {"Air Conditioner",      4,   2.0, 2, 18, 3}, // shiftable (type 2)
    {"Air Purifier",         6,   0.1, 2,  0, 3}, // shiftable
    {"Coffee Maker",         1,   0.4, 1,  7, 2}, // non-shiftable (morning), window +/-2
    {"Computer",             4,   0.25,1, 20, 3}, // non-shiftable (evening work)
    {"Digital Clock",       24,   0.0025,0,  0, 0}, // fixed: always on
    {"Dishwasher",           3,   1.33, 2, 21, 4}, // shiftable (evening ok)
    {"Hair Dryer",           1,   2.0, 1,  7, 2}, // non-shiftable (morning)
    {"Electric Iron",        1,   2.0, 1,  7, 2}, // non-shiftable (morning)
    {"EV Charger",           8,   4.0, 2, 22, 6}, // shiftable (night)
    {"Exhaust Fan",          2,   0.1, 2,  8, 3}, // shiftable
    {"Fan",                  8,   0.1, 2, 20, 4}, // shiftable
    {"Food Blender",         1,   0.4, 1,  8, 2}, // non-shiftable (morning)
    {"Induction Cooker",     1,   2.0, 1, 19, 2}, // non-shiftable (dinner)
    {"LED Lights",           5,   0.08,1, 19, 4}, // non-shiftable-ish (evening)
    {"Microwave",            1,   1.5, 1, 12, 2}, // non-shiftable (lunch)
    {"Night Light",          2,   0.05,0, 22, 0}, // fixed (night)
    {"Refrigerator",         6,   0.3, 0,  0, 0}, // fixed/cyclic (approx)
    {"Room Heater",          3,   2.0, 2,  6, 3}, // shiftable
    {"Router WiFi",         24,   0.025,0,  0, 0}, // fixed
    {"Shaver",               1,   0.05,1,  6, 2}, // non-shiftable
    {"Television",           3,   0.2, 1, 20, 3}, // non-shiftable (evening)
    {"Vacuum Cleaner",       1,   1.0, 2, 10, 8}, // shiftable (day)
    {"Washing Machine",      2,   0.6, 2,  9, 8}, // shiftable (day/night)
    {"Water Heater",         1,   2.5, 1,  6, 2}, // non-shiftable (morning)
    {"Water Pump",           1,   1.5, 1,  5, 2}  // non-shiftable (early morning)
};

// ---------- Helpers ----------
vector<int> schedule_from_start(int start, int dur) {
    vector<int> v(HOURS, 0);
    for (int i = 0; i < dur && start + i < HOURS; ++i) v[start + i] = 1;
    return v;
}

// returns allowable start range for appliance 'a'
pair<int,int> allowed_range_for_appliance(int aidx) {
    const Appliance &ap = appliances[aidx];
    int lo = 0, hi = HOURS - ap.duration;
    if (ap.type == 0) { // fixed
        lo = hi = ap.baseline;
    } else if (ap.type == 1) { // non-shiftable: restrict to baseline +/- window
        lo = max(0, ap.baseline - ap.window);
        hi = min(HOURS - ap.duration, ap.baseline + ap.window);
    } else { // shiftable: full range
        lo = 0; hi = HOURS - ap.duration;
    }
    return {lo, hi};
}

// baseline unscheduled starts (use appliances[].baseline for convenience)
vector<int> default_unscheduled_starts() {
    vector<int> s;
    s.reserve(appliances.size());
    for (size_t i = 0; i < appliances.size(); ++i) s.push_back(appliances[i].baseline);
    return s;
}

// compute hourly load from starts
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

// print schedule (start times)
void print_schedule(const vector<int>& starts) {
    for (size_t i = 0; i < appliances.size(); ++i) {
        cout << setw(2) << i << " : " << setw(20) << appliances[i].name
             << " start=" << setw(2) << starts[i]
             << " dur=" << setw(2) << appliances[i].duration
             << " type=" << (appliances[i].type==0?"fixed":appliances[i].type==1?"non-shift":"shift") << "\n";
    }
}

// mean, stdev, normalize helpers
double mean(const vector<double>& v){ double s=0; for(double x:v) s+=x; return s / (double)v.size(); }
double stdev(const vector<double>& v){ double m=mean(v); double s=0; for(double x:v) s+=(x-m)*(x-m); return sqrt(s / (double)v.size()); }
vector<double> normalize_vector(const vector<double>& v){
    double mn = *min_element(v.begin(), v.end()), mx = *max_element(v.begin(), v.end());
    vector<double> out(v.size());
    double denom = (mx - mn) + 1e-12;
    for (size_t i=0;i<v.size();++i) out[i] = (v[i] - mn) / denom;
    return out;
}

// ---------- Load limits and PL_obj generation (same formulas, with safeguards) ----------
struct LoadLimits { vector<double> PL_unsch, PL_N; double Plo, PLoff, PLon; };

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

    // Safeguards (as discussed)
    if (L.PLoff > L.PLon) swap(L.PLoff, L.PLon);
    if (L.PLoff < 0) L.PLoff = 0;
    if (L.PLoff > L.Plo) L.PLoff = L.Plo;
    if (L.PLon < L.Plo) L.PLon = L.Plo;
    return L;
}

vector<double> normalize_price(const vector<double>& tariff) {
    double mn = *min_element(tariff.begin(), tariff.end()), mx = *max_element(tariff.begin(), tariff.end());
    vector<double> p(HOURS);
    double denom = (mx - mn) + 1e-12;
    for (int t=0;t<HOURS;++t) p[t] = (tariff[t] - mn) / denom;
    return p;
}

vector<double> compute_PL_obj_from_price(const vector<double>& tariff, const LoadLimits& L, double K = 1.0) {
    vector<double> pnorm = normalize_price(tariff);
    vector<double> PLobj(HOURS, L.Plo);
    double mean_p = mean(pnorm);
    double span = (L.PLon - L.PLoff);
    double lo_bound = min(L.PLoff, L.PLon), hi_bound = max(L.PLoff, L.PLon);
    for (int t=0;t<HOURS;++t) {
        double delta = (mean_p - pnorm[t]) * span * K;
        double val = L.Plo + delta;
        if (val < lo_bound) val = lo_bound;
        if (val > hi_bound) val = hi_bound;
        PLobj[t] = val;
    }
    return PLobj;
}

// ---------- Load-shifting fitness F1 ----------
double load_shifting_fitness(
    const vector<int>& starts,
    const LoadLimits& limits,
    const vector<double>& PLobj,
    const vector<int>& off_peak_hours,
    const vector<int>& on_peak_hours,
    double penalty_weight = 1000.0
) {
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
            if (sched[t] < limits.PLoff) { double v = limits.PLoff - sched[t]; pen += v * v; }
        }
        if (is_on[t]) {
            if (sched[t] > limits.PLon)  { double v = sched[t] - limits.PLon; pen += v * v; }
        }
    }
    return sse + penalty_weight * pen;
}

// ---------- GA implementation (respects categories) ----------
using Chromosome = vector<vector<int>>;
const int GA_POP_SIZE   = 50;
const int GA_GENERATIONS= 200;
const double CROSSOVER_RATE = 0.85;
const double MUTATION_RATE  = 0.25;

// create random chromosome respecting categories & allowed ranges
Chromosome ga_createChromosome(const vector<int>& baselineStarts) {
    Chromosome chromosome; chromosome.reserve(appliances.size());
    for (size_t a = 0; a < appliances.size(); ++a) {
        auto range = allowed_range_for_appliance((int)a);
        int lo = range.first, hi = range.second;
        int start;
        if (appliances[a].type == 0) { // fixed
            start = baselineStarts[a];
        } else {
            if (hi < lo) hi = lo;
            start = lo + (rand() % (hi - lo + 1));
        }
        chromosome.push_back(schedule_from_start(start, appliances[a].duration));
    }
    return chromosome;
}

vector<int> ga_chromosome_to_starts(const Chromosome& ch) {
    vector<int> starts(ch.size(), 0);
    for (size_t a = 0; a < ch.size(); ++a) {
        int first = -1;
        for (int h = 0; h < HOURS; ++h) { if (ch[a][h] == 1) { first = h; break; } }
        if (first < 0) first = appliances[a].baseline;
        // enforce allowed range as safety
        auto rng = allowed_range_for_appliance((int)a);
        if (first < rng.first) first = rng.first;
        if (first > rng.second) first = rng.second;
        starts[a] = first;
    }
    return starts;
}

Chromosome ga_mutate(Chromosome ch, const vector<int>& baselineStarts) {
    if (((double) rand() / RAND_MAX) < MUTATION_RATE) {
        int idx = rand() % (int)appliances.size();
        // do not mutate fixed devices
        if (appliances[idx].type == 0) return ch;
        auto range = allowed_range_for_appliance(idx);
        int dur = appliances[idx].duration;
        int start = range.first + (rand() % (range.second - range.first + 1));
        ch[idx] = schedule_from_start(start, dur);
    }
    return ch;
}

Chromosome ga_crossover(const Chromosome &p1, const Chromosome &p2, const vector<int>& baselineStarts) {
    if (((double) rand() / RAND_MAX) < CROSSOVER_RATE) {
        int point = rand() % (int)appliances.size();
        Chromosome c1, c2;
        for (int i = 0; i < (int)appliances.size(); i++) {
            if (i < point) { c1.push_back(p1[i]); c2.push_back(p2[i]); }
            else           { c1.push_back(p2[i]); c2.push_back(p1[i]); }
        }
        // ensure fixed remain at baseline if crossover accidentally changed them
        for (size_t i = 0; i < appliances.size(); ++i) {
            if (appliances[i].type == 0) c1[i] = schedule_from_start(baselineStarts[i], appliances[i].duration);
            if (appliances[i].type == 0) c2[i] = schedule_from_start(baselineStarts[i], appliances[i].duration);
        }
        return c1;
    }
    // no crossover: return p1 (we will copy p1 to new population externally)
    return p1;
}

pair<vector<int>, double> run_GA_loadshift(
    const LoadLimits& limits,
    const vector<double>& PLobj,
    const vector<int>& off_peak_hours,
    const vector<int>& on_peak_hours,
    const vector<int>& baselineStarts,
    double penalty_weight = 1000.0,
    bool verbose = true
) {
    // init
    vector<Chromosome> population;
    population.reserve(GA_POP_SIZE);
    for (int i = 0; i < GA_POP_SIZE; ++i) population.push_back(ga_createChromosome(baselineStarts));

    Chromosome best = population[0];
    double bestFitness = load_shifting_fitness(ga_chromosome_to_starts(best), limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight);

    for (int gen = 0; gen < GA_GENERATIONS; gen++) {
        // compute fitnesses
        int P = (int)population.size();
        vector<double> fitnesses(P);
        for (int i = 0; i < P; ++i) fitnesses[i] = load_shifting_fitness(ga_chromosome_to_starts(population[i]), limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight);

        // selection weights (lower fitness -> higher weight)
        auto select_one = [&](void)->Chromosome {
            vector<double> weights(P);
            double eps = 1e-9;
            for (int i = 0; i < P; ++i) weights[i] = 1.0 / (fitnesses[i] + eps);
            double sum = accumulate(weights.begin(), weights.end(), 0.0);
            double pick = ((double) rand() / RAND_MAX) * sum;
            double cur = 0.0;
            for (int i = 0; i < P; ++i) { cur += weights[i]; if (cur >= pick) return population[i]; }
            return population.back();
        };

        vector<Chromosome> newpop;
        newpop.reserve(GA_POP_SIZE);
        // elitism: keep best
        for (int i = 0; i < P; ++i) if (fitnesses[i] < bestFitness) { bestFitness = fitnesses[i]; best = population[i]; }
        newpop.push_back(best);

        while ((int)newpop.size() < GA_POP_SIZE) {
            Chromosome p1 = select_one();
            Chromosome p2 = select_one();
            // crossover (we return single child to simplify)
            Chromosome c1 = ga_crossover(p1, p2, baselineStarts);
            Chromosome c2 = p2;
            // mutate but respect fixed / windows
            c1 = ga_mutate(c1, baselineStarts);
            c2 = ga_mutate(c2, baselineStarts);
            newpop.push_back(c1);
            if ((int)newpop.size() < GA_POP_SIZE) newpop.push_back(c2);
        }

        population.swap(newpop);

        for (auto &ch : population) {
            double f = load_shifting_fitness(ga_chromosome_to_starts(ch), limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight);
            if (f < bestFitness) { bestFitness = f; best = ch; }
        }
        if (verbose) cout << "GA Gen " << gen+1 << ": Best F1 = " << bestFitness << "\n";
    }

    vector<int> bestStarts = ga_chromosome_to_starts(best);
    return {bestStarts, bestFitness};
}

// ---------- AOA (respects fixed by clamping) ----------
struct Bounds { int lo, hi; };
vector<Bounds> aoa_bounds() {
    vector<Bounds> b; b.reserve(appliances.size());
    for (size_t i = 0; i < appliances.size(); ++i) {
        auto rng = allowed_range_for_appliance((int)i);
        b.push_back({rng.first, rng.second});
    }
    return b;
}
inline double urand_d() { return (double)rand() / (double)RAND_MAX; }
void discretize_round_clamp(const vector<double>& x, vector<int>& start, const vector<Bounds>& B) {
    start.resize(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        int v = (int)llround(x[i]);
        if (v < B[i].lo) v = B[i].lo;
        if (v > B[i].hi) v = B[i].hi;
        // enforce fixed
        if (appliances[i].type == 0) v = appliances[i].baseline;
        start[i] = v;
    }
}

const int AOA_POP = 50, AOA_ITER = 150;
const double C1 = 2.0, C2 = 6.0, C3 = 2.0, C4 = 1.0, U = 0.9, L = 0.1;

pair<vector<int>, double> run_AOA_loadshift(
    const vector<int>& seedStart,
    bool seedProvided,
    const LoadLimits& limits,
    const vector<double>& PLobj,
    const vector<int>& off_peak_hours,
    const vector<int>& on_peak_hours,
    double penalty_weight,
    bool verbose = true
) {
    int D = (int)appliances.size();
    vector<Bounds> B = aoa_bounds();

    vector<vector<double>> X(AOA_POP, vector<double>(D));
    vector<vector<double>> DEN(AOA_POP, vector<double>(D));
    vector<vector<double>> VOL(AOA_POP, vector<double>(D));
    vector<vector<double>> ACC(AOA_POP, vector<double>(D));

    auto rand_in_bounds = [&](int d){ return B[d].lo + urand_d() * (B[d].hi - B[d].lo); };

    for (int i = 0; i < AOA_POP; ++i) {
        for (int d = 0; d < D; ++d) {
            X[i][d] = rand_in_bounds(d);
            DEN[i][d] = urand_d();
            VOL[i][d] = urand_d();
            ACC[i][d] = urand_d();
        }
    }
    if (seedProvided) {
        for (int d = 0; d < D; ++d) X[0][d] = (double)seedStart[d];
    }

    double bestF = 1e100;
    vector<double> Xbest(D), DENbest(D), VOLbest(D), ACCbest(D);
    for (int i = 0; i < AOA_POP; ++i) {
        vector<int> s;
        discretize_round_clamp(X[i], s, B);
        double f = load_shifting_fitness(s, limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight);
        if (f < bestF) { bestF = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i]; }
    }

    vector<vector<double>> DENn(AOA_POP, vector<double>(D));
    vector<vector<double>> VOLn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCn(AOA_POP, vector<double>(D));
    vector<vector<double>> ACCnorm(AOA_POP, vector<double>(D));

    for (int t = 1; t <= AOA_ITER; ++t) {
        for (int i = 0; i < AOA_POP; ++i) for (int d = 0; d < D; ++d) {
            DENn[i][d] = DEN[i][d] + urand_d() * (DENbest[d] - DEN[i][d]);
            VOLn[i][d] = VOL[i][d] + urand_d() * (VOLbest[d] - VOL[i][d]);
        }

        double TF = exp((double(t) - AOA_ITER) / (double)AOA_ITER);
        double dens_decay = exp((double(AOA_ITER) - t) / (double)AOA_ITER) - (double)t / (double)AOA_ITER;

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

        for (int d = 0; d < D; ++d) {
            double mn = 1e100, mx = -1e100;
            for (int i = 0; i < AOA_POP; ++i) { mn = min(mn, ACCn[i][d]); mx = max(mx, ACCn[i][d]); }
            double denom = (mx - mn) + 1e-12;
            for (int i = 0; i < AOA_POP; ++i) {
                double z = (ACCn[i][d] - mn) / denom;
                ACCnorm[i][d] = U * z + L;
            }
        }

        for (int i = 0; i < AOA_POP; ++i) {
            if (TF <= 0.5) {
                int rr = rand() % AOA_POP;
                for (int d = 0; d < D; ++d) {
                    double step = C1 * urand_d() * ACCnorm[i][d] * dens_decay * (X[rr][d] - X[i][d]);
                    X[i][d] += step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            } else {
                double Tpar = C3 * TF;
                double P = 2.0 * urand_d() - C4;
                double F = (P <= 0.5) ? +1.0 : -1.0;
                for (int d = 0; d < D; ++d) {
                    double step = F * C2 * urand_d() * ACCnorm[i][d] * dens_decay * (Tpar * Xbest[d] - X[i][d]);
                    X[i][d] = Xbest[d] + step;
                    if (X[i][d] < B[d].lo) X[i][d] = B[d].lo;
                    if (X[i][d] > B[d].hi) X[i][d] = B[d].hi;
                }
            }
        }

        DEN.swap(DENn); VOL.swap(VOLn); ACC.swap(ACCn);
        for (int i = 0; i < AOA_POP; ++i) {
            vector<int> s; discretize_round_clamp(X[i], s, B);
            double f = load_shifting_fitness(s, limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight);
            if (f < bestF) { bestF = f; Xbest = X[i]; DENbest = DEN[i]; VOLbest = VOL[i]; ACCbest = ACC[i]; }
        }

        if (verbose) cout << "AOA Iter " << t << ": Best F1 = " << bestF << "\n";
    }

    vector<int> bestStart; discretize_round_clamp(Xbest, bestStart, aoa_bounds());
    return {bestStart, bestF};
}

// ---------- main ----------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    srand((unsigned)time(0));
    cout << fixed << setprecision(3);

    // baseline unscheduled & compute PL limits
    vector<int> baseStarts = default_unscheduled_starts();
    vector<double> PL_unsch = compute_hourly_load(baseStarts);
    LoadLimits limits = compute_load_limits_from_unscheduled(PL_unsch, 2.0);
    vector<double> PLobj = compute_PL_obj_from_price(tariff_default, limits, 1.0);

    cout << "Baseline (unscheduled) hourly load mean Plo = " << limits.Plo
         << ", PLoff = " << limits.PLoff << ", PLon = " << limits.PLon << "\n\n";

    cout << "Appliance categories (type: 0=fixed,1=non-shiftable,2=shiftable)\n";
    print_schedule(baseStarts);
    cout << "\nDesired PL_obj(t):\n";
    for (int h=0; h<HOURS; ++h) cout << setw(2) << h << ":" << PLobj[h] << "  ";
    cout << "\n\n";

    // off/on peak hours
    vector<int> off_peak_hours;
    for (int h=0; h<=6; ++h) off_peak_hours.push_back(h);
    off_peak_hours.push_back(23);
    vector<int> on_peak_hours = {18,19,20,21};

    // 1) GA
    double penalty_weight = 5000.0;
    auto ga_res = run_GA_loadshift(limits, PLobj, off_peak_hours, on_peak_hours, baseStarts, penalty_weight, true);
    vector<int> ga_best = ga_res.first;
    double ga_F = ga_res.second;

    cout << "\n=== GA Best (starts) ===\n"; print_schedule(ga_best);
    vector<double> ga_load = compute_hourly_load(ga_best);

    // 2) AOA seeded
    auto aoa_res = run_AOA_loadshift(ga_best, true, limits, PLobj, off_peak_hours, on_peak_hours, penalty_weight, true);
    vector<int> aoa_best = aoa_res.first;
    double aoa_F = aoa_res.second;

    cout << "\n=== AOA Best (starts final) ===\n"; print_schedule(aoa_best);
    vector<double> final_load = compute_hourly_load(aoa_best);

    cout << "\nHour, Baseline(kW), Desired PLobj(kW), Final Scheduled(kW), Delta(final-baseline)\n";
    for (int h=0; h<HOURS; ++h) {
        cout << setw(2) << h << " , " << setw(7) << PL_unsch[h]
             << " , " << setw(7) << PLobj[h]
             << " , " << setw(7) << final_load[h]
             << " , " << setw(7) << (final_load[h] - PL_unsch[h]) << "\n";
    }

    cout << "\nAppliance shift summary (baseline -> final)\n";
    for (size_t i=0;i<appliances.size();++i) {
        cout << setw(2) << i << " : " << setw(20) << appliances[i].name
             << "  base:" << setw(2) << appliances[i].baseline
             << "  final:" << setw(2) << aoa_best[i] << "  type:" << appliances[i].type << "\n";
    }

    cout << "\nDone.\n";
    return 0;
}
