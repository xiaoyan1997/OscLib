#ifndef OSCCALCANALYTIC_NSI_H
#define OSCCALCANALYTIC_NSI_H

// ============================================================================
// Why this class exists
// ---------------------
// Stan HMC needs the GRADIENT of the likelihood with respect to every fit
// parameter, which Stan gets by automatic differentiation (autodiff) through
// the oscillation calculator, with T = stan::math::var.  The exact NSI
// calculator, OscCalcPMNS_NSI, diagonalises the Hamiltonian numerically;
// under autodiff that cost ~0.97 s per likelihood evaluation.  This class
// computes the same probabilities in CLOSED FORM -- eigenvalues from Cardano's
// cubic formula, probabilities from a finite polynomial in the Hamiltonian --
// so autodiff is cheap: ~0.037 s per evaluation with systematics (~26x).
//
// It is upstream OscCalcAnalytic (unchanged, in OscCalcAnalytic.h/.cxx) with
// one physics change: the matter term of the Hamiltonian carries the NSI
// matrix epsilon (see _P in the .cxx).  It is the calculator behind the Stan
// HMC NSI productions from 8 July 2026 on (fit_stan_2024_analytic_NSI.C).
//
// Validation
//  * probabilities against OscCalcPMNS_NSI: test_nsi_all_channels.C, all
//    channels and phases, max |difference| ~ 1e-14 (rounding);
//  * autodiff gradients against finite differences: min_grad_test.C
//    (AD = FD), cubic_grad_test.C, and test_cubic_grad_from_nsi_cxx below
//    (in the .cxx).
//  * History: the first runs (7-8 July) saw ZERO epsilon gradients.  The
//    cause was a stale libOscLib.so that had not been rebuilt after this class
//    was added -- not the mathematics.  After the rebuild AD matched FD.
// ============================================================================

// OscCalcAnalytic_NSI: NSI-capable oscillation calculator based on OscCalcAnalytic.
// Adds the full 9-parameter NSI matter Hamiltonian (Cardano eigensolver).
// Internal NSI storage: polar (magnitude + phase) matching OscCalcPMNS_NSI API
// so FitVarsNSI dispatch code can use both calculators interchangeably.
// Placed in osc::analytic namespace; class name _OscCalcNSI<T>.

#include "OscLib/OscCalcAnalytic.h"   // pulls in cmplx, Hermitian, Eigenvalues, ProbCache, etc.

namespace osc::analytic
{
  template<class T> class _OscCalcNSI: public _IOscCalcAdjustable<T>,
                                       protected ProbCache<double, T>,
                                       protected ProbCache<Eigen::ArrayXd, Eigen::ArrayX<T>>
  {
  public:
    _OscCalcNSI();
    virtual ~_OscCalcNSI();
    using _IOscCalc<T>::P;

    virtual _IOscCalcAdjustable<T>* Copy() const override;

    virtual void SetL(double L) override;
    virtual void SetRho(double rho) override;
    virtual void SetDmsq21(const T& dmsq21) override;
    virtual void SetDmsq32(const T& dmsq32) override;
    virtual void SetTh12(const T& th12) override;
    virtual void SetTh13(const T& th13) override;
    virtual void SetTh23(const T& th23) override;
    virtual void SetdCP(const T& dCP) override;

    virtual T P(int from, int to, double E) override;
    virtual Eigen::ArrayX<T> P(int from, int to, const std::vector<double>& E) override;
    virtual Eigen::ArrayX<T> P(int from, int to, const Eigen::ArrayXd& E) override;

    virtual TMD5* GetParamsHash() const override;

    // Units and conventions (same as OscCalcPMNS_NSI, so FitVarsNSI can use
    // either calculator): epsilons are dimensionless, in units of the
    // charged-current potential V_CC; phases in RADIANS (the fit variables
    // convert from units of pi).  Each setter only clears the probability
    // cache: epsilon enters the matter part of the Hamiltonian, which is
    // rebuilt in _P on every uncached call anyway.
    // NSI parameters — polar storage, matching OscCalcPMNS_NSI API.
    // Diagonal epsilons are real (no phase).
    // Off-diagonal: magnitude (eps) + phase (delta).
    void SetEps_ee     (const T& v) { fEps_ee      = v;       ClearProbCaches(); }
    void SetEps_mumu   (const T& v) { fEps_mumu    = v;       ClearProbCaches(); }
    void SetEps_tautau (const T& v) { fEps_tautau  = v;       ClearProbCaches(); }
    void SetEps_emu    (const T& v) { fEps_emu     = v;       ClearProbCaches(); }
    void SetEps_etau   (const T& v) { fEps_etau    = v;       ClearProbCaches(); }
    void SetEps_mutau  (const T& v) { fEps_mutau   = v;       ClearProbCaches(); }
    void SetDelta_emu  (const T& v) { fDelta_emu   = v;       ClearProbCaches(); }
    void SetDelta_etau (const T& v) { fDelta_etau  = v;       ClearProbCaches(); }
    void SetDelta_mutau(const T& v) { fDelta_mutau = v;       ClearProbCaches(); }

    T GetEps_ee()      const { return fEps_ee; }
    T GetEps_mumu()    const { return fEps_mumu; }
    T GetEps_tautau()  const { return fEps_tautau; }
    T GetEps_emu()     const { return fEps_emu; }
    T GetEps_etau()    const { return fEps_etau; }
    T GetEps_mutau()   const { return fEps_mutau; }
    T GetDelta_emu()   const { return fDelta_emu; }
    T GetDelta_etau()  const { return fDelta_etau; }
    T GetDelta_mutau() const { return fDelta_mutau; }

    // Cartesian convenience setters (Re/Im → magnitude/phase conversion).
    // Useful for test scripts and initialization.
    // NOTE: these take plain doubles, so they are for tests and starting
    // values only; nothing differentiable should go through them.
    void SetEps_emu_cart  (double re, double im) {
      fEps_emu = std::sqrt(re*re + im*im);
      fDelta_emu = std::atan2(im, re);
      ClearProbCaches();
    }
    void SetEps_etau_cart (double re, double im) {
      fEps_etau = std::sqrt(re*re + im*im);
      fDelta_etau = std::atan2(im, re);
      ClearProbCaches();
    }
    void SetEps_mutau_cart(double re, double im) {
      fEps_mutau = std::sqrt(re*re + im*im);
      fDelta_mutau = std::atan2(im, re);
      ClearProbCaches();
    }

  protected:
    void ClearProbCaches()
    {
      ProbCache<double, T>::clear();
      ProbCache<Eigen::ArrayXd, Eigen::ArrayX<T>>::clear();
    }

    template<class VT, class KVT> VT _P(int from, int to, const KVT& E);

    // Caching, inherited from upstream OscCalcAnalytic: the PMNS elements and
    // the vacuum Hamiltonian are recomputed only when an angle or mass
    // splitting changed ("dirty"); probabilities per energy are cached in the
    // ProbCache bases until any parameter changes.
    bool fDirty12, fDirty13, fDirty23, fDirtyCP, fDirtyMasses;

    T s12, c12, s13, c13, s23, c23, sCP, cCP;

    /*T        Ue1;*/ T        Ue2; cmplx<T>  Ue3;
    /*cmplx<T> Um1;*/ cmplx<T> Um2; T         Um3;
    /*cmplx<T> Ut1;*/ cmplx<T> Ut2; T         Ut3;

    inline __attribute__((always_inline)) void UpdatePMNS();

    // VACUUM Hamiltonian in the flavour basis, times 2E (units eV^2):
    //   H = U diag(0, dm21, dm31) U^dagger.  Hermitian, so 3 real diagonal and
    //   3 complex upper-triangle elements suffice.  The matter + NSI part is
    //   added in _P, because it does not scale with 1/E.
    T Hee;            cmplx<T>  Hem; cmplx<T>  Het;
    /*cmplx<T> Hme;*/ T         Hmm; cmplx<T>  Hmt;
    /*cmplx<T> Hte; cmplx<T>  Htm;*/ T         Htt;

    inline __attribute__((always_inline)) void UpdateHamiltonian();
    inline __attribute__((always_inline)) double Hmat();

    // Stored as type T (not double): with T = stan::math::var every epsilon is
    // a node on Stan's autodiff tape, which is what gives HMC its gradients.
    // NSI epsilon parameters — polar storage, type T so Stan gradients flow through.
    T fEps_ee      = T(0);
    T fEps_mumu    = T(0);
    T fEps_tautau  = T(0);
    T fEps_emu     = T(0);   T fDelta_emu    = T(0);
    T fEps_etau    = T(0);   T fDelta_etau   = T(0);
    T fEps_mutau   = T(0);   T fDelta_mutau  = T(0);

  private:
    _OscCalcNSI(const _OscCalcNSI&) = default;
    _OscCalcNSI& operator=(const _OscCalcNSI&) = default;
  };
} // namespace osc::analytic

namespace osc
{
  template<class T> using _OscCalcAnalytic_NSI = osc::analytic::_OscCalcNSI<T>;
  using OscCalcAnalytic_NSI = _OscCalcAnalytic_NSI<double>;

  // Convenience downcast — mirrors DowncastToNSI for PMNS_NSI.
  template<class T>
  osc::analytic::_OscCalcNSI<T>* DowncastToAnalyticNSI(_IOscCalc<T>* calc) {
    return dynamic_cast<osc::analytic::_OscCalcNSI<T>*>(calc);
  }
  template<class T>
  const osc::analytic::_OscCalcNSI<T>* DowncastToAnalyticNSI(const _IOscCalc<T>* calc) {
    return dynamic_cast<const osc::analytic::_OscCalcNSI<T>*>(calc);
  }
}

#endif
