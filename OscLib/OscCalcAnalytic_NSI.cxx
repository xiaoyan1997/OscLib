#ifdef OSCLIB_STAN
#include "OscLib/Stan.h"
#endif
// ============================================================================
// OscCalcAnalytic_NSI.cxx -- closed-form three-flavour oscillation
// probabilities in matter WITH NSI, differentiable by Stan autodiff.
//
// Reading order:
//   1. _P (near the end): the physics.  Builds M = -L * H for the full
//      Hamiltonian (vacuum + matter + NSI), then the probabilities.
//   2. Hermitian<T>::GetEigenvalues + SolveCubicNSI: eigenvalues of M in
//      closed form, and the sums the probability formula needs.
//   3. Setters, UpdatePMNS, UpdateHamiltonian: the vacuum part, unchanged
//      from upstream OscCalcAnalytic.
//   4. The block of small templates at the top only re-declares helpers that
//      upstream keeps inside OscCalcAnalytic.cxx (see the comments there).
//
// The physics in one line:
//   H = (1/2E) U diag(0, dm21, dm31) U^dagger  +  V_CC [ diag(1,0,0) + eps ],
//   V_CC = sqrt(2) G_F N_e,   eps = Hermitian 3x3,  eps_ab = |eps_ab| e^{i delta_ab},
//   amplitude S = exp(-i H L),   P(a -> b) = |S_ba|^2.
// Standard oscillations are eps = 0, where this reduces to upstream exactly.
// ============================================================================


#include "OscLib/OscCalcAnalytic_NSI.h"

#include <array>
#include <cassert>

#include "OscLib/Constants.h"

// Why the helpers below are duplicated: upstream defines sincos(var), the cmplx
// operators, the cubic solver and GetEigenvalues inside OscCalcAnalytic.cxx,
// not in a header, so this separate translation unit cannot see them and must
// provide its own.  They are templates (weak symbols), and the ones that could
// clash by name are renamed (sqrN, cubeN, SolveCubicNSI, kSqrt3_NSI).
// Stan sincos — defined once in OscCalcAnalytic.cxx; forward-declare here.
#ifdef OSCLIB_STAN
void sincos(const stan::math::var& x, stan::math::var* sx, stan::math::var* cx);
#endif

// Vector sincos helper — template, OK to re-define across translation units.
template<class T, class U> void sincos(T& x,
                                       Eigen::ArrayX<U>* sx,
                                       Eigen::ArrayX<U>* cx)
{
  sx->resize(x.size());
  cx->resize(x.size());
  for(int i = 0; i < x.size(); ++i) sincos(x[i], &(*sx)[i], &(*cx)[i]);
}

namespace osc::analytic
{
  // cmplx arithmetic operators — defined in OscCalcAnalytic.cxx but NOT in any
  // header, so they must be duplicated here. All templates → weak symbols → safe.
  template<class T, class U> cmplx<T, U> make_cmplx(const T& re, const U& im){return cmplx<T, U>(re, im);}

  template<class A, class B, class C, class D> inline __attribute__((always_inline)) auto
  operator*(const cmplx<A, B>& x, const cmplx<C, D>& y)
  {
    return make_cmplx(x.re*y.re - x.im*y.im, x.re*y.im + x.im*y.re);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator*(const A& x, const cmplx<B, C>& y)
  {
    return make_cmplx(x*y.re, x*y.im);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator*(const cmplx<A, B>& x, const C& y)
  {
    return make_cmplx(x.re*y, x.im*y);
  }

  template<class T, class U> inline __attribute__((always_inline)) auto
  operator/(const cmplx<T>& x, const U& y)
  {
    return cmplx<T>(x.re/y, x.im/y);
  }

  template<class A, class B, class C, class D> inline __attribute__((always_inline)) auto
  operator+(const cmplx<A, B>& x, const cmplx<C, D>& y)
  {
    return make_cmplx(x.re + y.re, x.im + y.im);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator+(const cmplx<A, B>& x, const C& y)
  {
    return make_cmplx(x.re + y, x.im);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator+(const A& x, const cmplx<B, C>& y)
  {
    return make_cmplx(x + y.re, y.im);
  }

  template<class A, class B, class C, class D> inline __attribute__((always_inline)) auto
  operator-(const cmplx<A, B>& x, const cmplx<C, D>& y)
  {
    return make_cmplx(x.re - y.re, x.im - y.im);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator-(const cmplx<A, B>& x, const C& y)
  {
    return make_cmplx(x.re - y, x.im);
  }

  template<class A, class B, class C> inline __attribute__((always_inline)) auto
  operator-(const A& x, const cmplx<B, C>& y)
  {
    return make_cmplx(x - y.re, -y.im);
  }

  template<class T> inline __attribute__((always_inline)) cmplx<T>
  operator-(const cmplx<T>& x)
  {
    return make_cmplx(-x.re, -x.im);
  }

  // sqr/cube — template weak symbols, OK to re-define.
  template<class T> T sqrN(T x){return x*x;}
  template<class T> T cubeN(T x){return x*x*x;}

  // Use a static constexpr instead of const double sqrt3 to avoid ODR conflict
  // with the same-named variable in OscCalcAnalytic.cxx.
  static constexpr double kSqrt3_NSI = 1.7320508075688772935;

  // ---------------------------------------------------------------------------
  // Cardano's TRIGONOMETRIC solution of a cubic with three real roots.
  // M is Hermitian, so its characteristic polynomial always has three real
  // roots; in that case p < 0 and the roots are
  //   x_k = 2 sqrt(-p) cos( (1/3) acos( q / (2 p sqrt(-p)) ) - 2 pi k / 3 ) - b/3,
  // with p and q the depressed-cubic coefficients computed in the first lines.
  // (t0, t1, t2 below are those three cosines written with sin/cos of one
  // angle r; t2 uses that the shifted roots sum to zero.)  Every operation is
  // a smooth function -- sqrt, acos, sincos -- so autodiff differentiates it
  // directly, which is the reason for using the closed form at all.
  // Same algorithm as upstream's solver.
  /// Solve x^3 + b*x^2 + c*x + d = 0  (NSI version uses sqrN/cubeN to avoid ODR)
  template<class T> std::array<T, 3> SolveCubicNSI(T b, T c, T d)
  {
    b /= 3;
    const T p = c/3 - sqrN(b);
    const T q = 2*cubeN(b) - b*c + d;

    const T s = sqrt(-p);
    const T r = acos(q/(2*p*s)) / 3;

    T sinr, cosr;
    sincos(r, &sinr, &cosr);

    const T t0 = 2*s*cosr;
    const T t1 = s*(kSqrt3_NSI*sinr - cosr);
    const T t2 = -t0-t1;

    return {t0 - b, t1 - b, t2 - b};
  }

  //---------------------------------------------------------------------------
  // GetEigenvalues must be defined here because _P() instantiates it,
  // and its definition in OscCalcAnalytic.cxx is not visible in this TU.
  // Template method → weak symbol → linker picks one; both implementations
  // are mathematically identical.
  // Eigenvalues of M from its characteristic polynomial
  //   det(x I - M) = x^3 + b x^2 + c x + d,
  //   b = -trace M,  c = sum of the three principal 2x2 minors,  d = -det M,
  // written out for a Hermitian M (|M_em|^2 etc. are the .norm() terms).
  //
  // It then returns the three sums the probability formula needs:
  //   sume   = sum_k  e^{i x_k}        / p'(x_k)
  //   sumxe  = sum_k  x_k e^{i x_k}    / p'(x_k)
  //   sumxxe = sum_k  x_k^2 e^{i x_k}  / p'(x_k),     p'(x) = 3x^2 + 2bx + c,
  // with the common phase e^{i x_0} factored out (it cancels in |amplitude|^2).
  // These come from Sylvester's formula for a matrix function:
  //   exp(iM) = sum_k e^{i x_k} adj(x_k I - M) / p'(x_k),
  // and adj(x I - M) is a quadratic polynomial in x whose coefficients are
  // entries and cofactors of M -- that is the combination used in _P.
  template<class T> Eigenvalues<T> Hermitian<T>::GetEigenvalues()
  {
    const auto& M = *this;

    const T b = -M.ee-M.mm-M.tt;
    const T Lem = M.em.norm();
    const T Let = M.et.norm();
    const T Lmt = M.mt.norm();
    const T c =  M.ee*M.mm + M.ee*M.tt + M.mm*M.tt - Lem - Let - Lmt;
    const T d = (M.ee*Lmt + M.mm*Let + M.tt*Lem
                 - M.ee*M.mm*M.tt
                 -2*(M.em.re * M.mt.re * M.et.re + M.em.re * M.mt.im * M.et.im
                   + M.em.im * M.mt.re * M.et.im - M.em.im * M.mt.im * M.et.re));

    const std::array<T, 3> xs = SolveCubicNSI(b, c, d);

    T c10, s10, c20, s20;
    sincos(xs[1]-xs[0], &s10, &c10);
    sincos(xs[2]-xs[0], &s20, &c20);

    const T ei0 = 1/(3*sqrN(xs[0]) + 2*b*xs[0] + c);
    const cmplx<T> ei1 = cmplx(c10, s10)/(3*sqrN(xs[1]) + 2*b*xs[1] + c);
    const cmplx<T> ei2 = cmplx(c20, s20)/(3*sqrN(xs[2]) + 2*b*xs[2] + c);

    return {
      ei0 + ei1 + ei2,
      xs[0]*ei0 + xs[1]*ei1 + xs[2]*ei2,
      sqrN(xs[0])*ei0 + sqrN(xs[1])*ei1 + sqrN(xs[2])*ei2};
  }

  //---------------------------------------------------------------------------
  template<class T> _OscCalcNSI<T>::_OscCalcNSI()
    : fDirty12(true), fDirty13(true), fDirty23(true), fDirtyCP(true), fDirtyMasses(true),
      Ue3(0, 0), Um2(0, 0), Ut2(0, 0),
      Hem(0, 0), Het(0, 0), Hmt(0, 0)
  {
  }

  //---------------------------------------------------------------------------
  template<class T> _OscCalcNSI<T>::~_OscCalcNSI()
  {
  }

  //---------------------------------------------------------------------------
  template<class T> _IOscCalcAdjustable<T>* _OscCalcNSI<T>::Copy() const
  {
    return new _OscCalcNSI<T>(*this);
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetL(double L)
  {
    if(L == this->fL) return;
    this->fL = L;
    ClearProbCaches();
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetRho(double rho)
  {
    if(rho == this->fRho) return;
    this->fRho = rho;
    ClearProbCaches();
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetDmsq21(const T& dmsq21)
  {
    if constexpr(std::is_arithmetic_v<T>) if(dmsq21 == this->fDmsq21) return;
    this->fDmsq21 = dmsq21;
    fDirtyMasses = true;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetDmsq32(const T& dmsq32)
  {
    if constexpr(std::is_arithmetic_v<T>) if(dmsq32 == this->fDmsq32) return;
    this->fDmsq32 = dmsq32;
    fDirtyMasses = true;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetTh23(const T& th23)
  {
    if constexpr(std::is_arithmetic_v<T>) if(th23 == this->fTh23) return;
    this->fTh23 = th23;
    fDirty23 = true;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetTh13(const T& th13)
  {
    if constexpr(std::is_arithmetic_v<T>) if(th13 == this->fTh13) return;
    this->fTh13 = th13;
    fDirty13 = true;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetTh12(const T& th12)
  {
    if constexpr(std::is_arithmetic_v<T>) if(th12 == this->fTh12) return;
    this->fTh12 = th12;
    fDirty12 = true;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::SetdCP(const T& delta)
  {
    if constexpr(std::is_arithmetic_v<T>) if(delta == this->fdCP) return;
    this->fdCP = delta;
    fDirtyCP = true;
  }

  //---------------------------------------------------------------------------
  template<class T> TMD5* _OscCalcNSI<T>::GetParamsHash() const
  {
    return _IOscCalcAdjustable<T>::GetParamsHashDefault("AnalyticNSI");
  }

  //---------------------------------------------------------------------------
  template<class T> double _OscCalcNSI<T>::Hmat()
  {
    return constants::kMatterDensityToEffect * this->fRho * constants::kZPerA;
  }

  //---------------------------------------------------------------------------
  template<class T> void _OscCalcNSI<T>::UpdatePMNS()
  {
    Ue2 = s12*c13;
    Um3 = s23*c13;
    Ut3 = c23*c13;

    const cmplx<T> phase(cCP, sCP);
    Ue3 = s13*phase.conj();
    Um2 =  c12*c23-(s12*s23*s13)*phase;
    Ut2 = -c12*s23-(s12*c23*s13)*phase;
  }

  //---------------------------------------------------------------------------
  // Vacuum Hamiltonian (times 2E) in the flavour basis:
  //   H_ab = dm21 U_a2 U_b2^*  +  dm31 U_a3 U_b3^*     (the m1 = 0 column drops out).
  // Unchanged from upstream.
  template<class T> void _OscCalcNSI<T>::UpdateHamiltonian()
  {
    const T d2 = this->fDmsq21;
    const T d3 = this->fDmsq21 + this->fDmsq32;

    Hee = d2 * sqrN(Ue2)   + d3 * Ue3.norm();
    Hmm = d2 * Um2.norm() + d3 * sqrN(Um3);
    Htt = d2 * Ut2.norm() + d3 * sqrN(Ut3);

    Hem = d2 * Ue2 * Um2.conj() + d3 * Ue3 * Um3;
    Het = d2 * Ue2 * Ut2.conj() + d3 * Ue3 * Ut3;
    Hmt = d2 * Um2 * Ut2.conj() + d3 * Um3 * Ut3;

    ClearProbCaches();
  }

  //---------------------------------------------------------------------------
  template<class T> template<class VT, class KVT> VT _OscCalcNSI<T>::
  _P(int from, int to, const KVT& E)
  {
    // Antineutrinos (negative PDG codes): evaluate the neutrino formula at -E.
    // That flips the sign of the vacuum term only, giving M' = -conj(M_anti),
    // where M_anti is the true antineutrino matrix (U -> U*, V -> -V,
    // eps -> eps*).  Then exp(iM') = conj(exp(i M_anti)), and the
    // probabilities |.|^2 are identical.  So the NSI phases are conjugated
    // correctly for antineutrinos with no extra code (commit 4a6aa43).
    if(from < 0) return P(-from, -to, -E);


    assert(from > 0 && to > 0);
    assert(from == 12 || from == 14 || from == 16);
    assert(to == 12 || to == 14 || to == 16);

    const bool dirtyAngles = fDirty12 || fDirty13 || fDirty23 || fDirtyCP;

    if(dirtyAngles){
      if(fDirty12) sincos(this->fTh12, &s12, &c12);
      if(fDirty13) sincos(this->fTh13, &s13, &c13);
      if(fDirty23) sincos(this->fTh23, &s23, &c23);
      if(fDirtyCP) sincos(this->fdCP,  &sCP, &cCP);
      UpdatePMNS();
      UpdateHamiltonian();
    }
    else{
      if(fDirtyMasses){
        UpdateHamiltonian();
      }
      else{
        auto it = ProbCache<KVT, VT>::find(E);
        if(it != ProbCache<KVT, VT>::end()) return it->second.P(from, to);
      }
    }

    fDirty12 = fDirty13 = fDirty23 = fDirtyCP = fDirtyMasses = false;

    // Work with the dimensionless matrix M = -L * H (so that S = exp(iM)).
    // k = -L / (2E) with L in km, E in GeV, converted so that (eV^2) * k is a
    // pure number:  vacuum part = H_vac(2E) * k.
    const KVT k = (constants::kkmTom / (constants::kInversemToeV * constants::kGeVToeV * 2) * -this->fL) / E;
    Hermitian<VT> M;
    // NSI matter Hamiltonian: H_mat += A_cc * eps_ab.
    // Off-diagonal eps stored as polar (magnitude + phase); convert to Cartesian here.
    // A_nsi = A_cc * L in units compatible with k above.
    // A_nsi = L * V_CC, the matter potential (Hmat = sqrt(2) G_F N_e from the
    // density rho and Z/A = 0.5) times the baseline, in the same units.  It is
    // the same number upstream subtracts from M.ee; the name only says that it
    // now also multiplies the NSI matrix.
    const double A_nsi = this->fL * constants::kkmTom / constants::kInversemToeV * Hmat();
    // Polar → Cartesian for off-diagonal epsilons.
    T sde, cde, see, cee, smt, cmt;
    sincos(fDelta_emu,   &sde, &cde);
    sincos(fDelta_etau,  &see, &cee);
    sincos(fDelta_mutau, &smt, &cmt);
    // THE NSI CHANGE.  Upstream has only M.ee = Hee*k - A_nsi (the standard
    // MSW term, V_CC on the ee entry).  With NSI the matter term is
    // V_CC (diag(1,0,0) + eps), so every entry gets -A_nsi * eps_ab:
    //   diagonal:      eps_ee (added to the 1), eps_mumu, eps_tautau  (real)
    //   off-diagonal:  eps_ab = |eps_ab| (cos delta_ab + i sin delta_ab)
    // Only the upper triangle (em, et, mt) is stored; M is Hermitian.
    M.ee = Hee * k  - A_nsi * (1.0 + fEps_ee);
    M.em = Hem * k  - A_nsi * cmplx<T>(fEps_emu   * cde, fEps_emu   * sde);
    M.mm = Hmm * k  - A_nsi * fEps_mumu;
    M.et = Het * k  - A_nsi * cmplx<T>(fEps_etau  * cee, fEps_etau  * see);
    M.mt = Hmt * k  - A_nsi * cmplx<T>(fEps_mutau * cmt, fEps_mutau * smt);
    M.tt = Htt * k  - A_nsi * fEps_tautau;

    // Amplitudes from Sylvester's formula (see GetEigenvalues):
    //   S_ab ~ cofactor_ab * sume  +/-  (linear in M) * sumxe  (+ sumxxe on the diagonal).
    // Aee, Amm, Aem are the needed 2x2 cofactors of M.  Probs takes the four
    // independent probabilities P(e->e), P(mu->e), P(e->mu), P(mu->mu) and
    // fills the other five by unitarity (Probs in Cache.h).
    const Eigenvalues<VT> es = M.GetEigenvalues();
    const VT Aee = M.mm*M.tt - M.mt.norm();
    const VT Amm = M.ee*M.tt - M.et.norm();
    const cmplx<VT> Aem = M.et*M.mt.conj() - M.em*M.tt;

    const Probs<VT> ps((Aee       *es.sume - (M.mm+M.tt) *es.sumxe + es.sumxxe).norm(),
                       (Aem       *es.sume +  M.em       *es.sumxe            ).norm(),
                       (Aem.conj()*es.sume +  M.em.conj()*es.sumxe            ).norm(),
                       (Amm       *es.sume - (M.ee+M.tt) *es.sumxe + es.sumxxe).norm());

    // Cache per energy until a parameter changes.
    ProbCache<KVT, VT>::emplace(E, ps);

    return ps.P(from, to);
  }

  //---------------------------------------------------------------------------
  template<class T> T _OscCalcNSI<T>::P(int from, int to, double E)
  {
    return _P<T>(from, to, E);
  }

  //---------------------------------------------------------------------------
  template<class T> Eigen::ArrayX<T> _OscCalcNSI<T>::
  P(int from, int to, const Eigen::ArrayXd& E)
  {
    return _P<Eigen::Array<T, Eigen::Dynamic, 1>>(from, to, E);
  }

  //---------------------------------------------------------------------------
  template<class T> Eigen::ArrayX<T> _OscCalcNSI<T>::
  P(int from, int to, const std::vector<double>& E)
  {
    return P(from, to, Eigen::Map<const Eigen::ArrayXd>(E.data(), E.size()));
  }

} // namespace osc::analytic


// double: used by ARIA/tempering (no gradients).  stan::math::var: used by
// Stan HMC (gradients via autodiff); compiled only in OSCLIB_STAN builds.
// Explicit instantiations
template class osc::analytic::_OscCalcNSI<double>;

#ifdef OSCLIB_STAN
template class osc::analytic::_OscCalcNSI<stan::math::var>;
#endif

#ifdef OSCLIB_STAN
// Kept from the 8 July gradient investigation: compares the autodiff derivative
// of a cubic root with a central finite difference, from inside this TU.
// Test helper: calls SolveCubicNSI<var> from within this TU to check gradient propagation.
// Exposed as a plain C++ function callable from test macros.
extern "C" {
void test_cubic_grad_from_nsi_cxx(double eps_in, double* g_ad, double* g_fd)
{
  using var = stan::math::var;
  // b, c, d that depend linearly on eps_in (simple known gradient)
  var eps_v(eps_in);
  var b = eps_v * var(1.0) + var(-0.5);
  var c = eps_v * var(0.3) + var(0.2);
  var d = eps_v * var(-0.1) + var(0.05);
  auto xs = osc::analytic::SolveCubicNSI(b, c, d);
  // Use xs[0] as the output
  stan::math::grad(xs[0].vi_);
  *g_ad = eps_v.adj();
  stan::math::recover_memory();

  // FD reference (double arithmetic)
  double h = 1e-5;
  {
    double bh = (eps_in+h) * 1.0 + (-0.5);
    double ch = (eps_in+h) * 0.3 + 0.2;
    double dh = (eps_in+h) * (-0.1) + 0.05;
    std::array<double,3> xh = osc::analytic::SolveCubicNSI(bh, ch, dh);
    double bl = (eps_in-h) * 1.0 + (-0.5);
    double cl = (eps_in-h) * 0.3 + 0.2;
    double dl = (eps_in-h) * (-0.1) + 0.05;
    std::array<double,3> xl = osc::analytic::SolveCubicNSI(bl, cl, dl);
    *g_fd = (xh[0] - xl[0]) / (2*h);
  }
}
} // extern "C"
#endif
