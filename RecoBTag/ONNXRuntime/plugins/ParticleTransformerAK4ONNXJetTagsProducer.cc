#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using namespace cms::Ort;

class ParticleTransformerAK4ONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit ParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~ParticleTransformerAK4ONNXJetTagsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef std::vector<reco::ParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(btagbtvdeep::ParticleTransformerAK4Features features);
  void get_input_sizes(const reco::FeaturesTagInfo<btagbtvdeep::ParticleTransformerAK4Features> taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  // Modified for 4 inputs instead of 6
  enum InputIndexes {
    kGlobalFeatures = 0,
    kChargedCandidates = 1,
    kNeutralCandidates = 2,
    kVertices = 3
  };
  
  // Global features
  constexpr static unsigned n_features_global_ = 17;
  
  // CPF features
  unsigned n_cpf_;
  constexpr static unsigned n_features_cpf_ = 31;  // Including 4-vec at the end
  
  // NPF features  
  unsigned n_npf_;
  constexpr static unsigned n_features_npf_ = 14;  // Including 4-vec at the end
  
  // SV features
  unsigned n_sv_;
  constexpr static unsigned n_features_sv_ = 19;  // Including 4-vec at the end
  
  std::vector<unsigned> input_sizes_;
  std::vector<std::vector<int64_t>> input_shapes_;

  // hold the input data
  FloatArrays data_;
};

ParticleTransformerAK4ONNXJetTagsProducer::ParticleTransformerAK4ONNXJetTagsProducer(const edm::ParameterSet& iConfig,
                                                                                     const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")) {
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }
}

void ParticleTransformerAK4ONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfParticleTransformerAK4ChargeJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfParticleTransformerAK4TagInfos"));
  desc.add<std::vector<std::string>>("input_names", {"global_features", "cpf_features", "npf_features", "vtx_features"});
  desc.add<edm::FileInPath>("model_path",
                            edm::FileInPath("RecoBTag/Combined/data/models/particletransformer_AK4/1/full_model_best.onnx"));
  desc.add<std::vector<std::string>>("output_names", {"output"});
  desc.add<std::vector<std::string>>(
      "flav_names", std::vector<std::string>{"probb", "probbb", "problepb", "probc", "probuds", "probg", 
                                              "probchargeneg", "probchargepos", "probchargezero"});

  descriptions.add("pfParticleTransformerAK4ChargeJetTags", desc);
}

std::unique_ptr<ONNXRuntime> ParticleTransformerAK4ONNXJetTagsProducer::initializeGlobalCache(
    const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void ParticleTransformerAK4ONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void ParticleTransformerAK4ONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  // initialize output collection
  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  if (!tag_infos->empty()) {
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }
  } else {
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
    const auto& taginfo = (*tag_infos)[jet_n];
    std::vector<float> outputs(flav_names_.size(), -1.0);
    if (taginfo.features().is_filled) {
      get_input_sizes(taginfo);

      // run prediction with dynamic batch size per event
      input_shapes_ = {{(int64_t)1, (int64_t)n_features_global_},           // global features
                       {(int64_t)1, (int64_t)n_cpf_, (int64_t)n_features_cpf_},  // cpf features  
                       {(int64_t)1, (int64_t)n_npf_, (int64_t)n_features_npf_},  // npf features
                       {(int64_t)1, (int64_t)n_sv_, (int64_t)n_features_sv_}};   // vtx features

      outputs = globalCache()->run(input_names_, data_, input_shapes_, output_names_, 1)[0];
      assert(outputs.size() == flav_names_.size());
    }

    const auto& jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
      (*(output_tags[flav_n]))[jet_ref] = outputs[flav_n];
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void ParticleTransformerAK4ONNXJetTagsProducer::get_input_sizes(
    const reco::FeaturesTagInfo<btagbtvdeep::ParticleTransformerAK4Features> taginfo) {
  const auto& features = taginfo.features();

  unsigned int n_cpf = features.c_pf_features.size();
  unsigned int n_npf = features.n_pf_features.size();
  unsigned int n_vtx = features.sv_features.size();

  n_cpf_ = std::max((unsigned int)1, n_cpf);
  n_npf_ = std::max((unsigned int)1, n_npf);
  n_sv_ = std::max((unsigned int)1, n_vtx);

  n_cpf_ = std::min((unsigned int)26, n_cpf_);
  n_npf_ = std::min((unsigned int)25, n_npf_);
  n_sv_ = std::min((unsigned int)5, n_sv_);
  
  input_sizes_ = {
      n_features_global_,                // global features
      n_cpf_ * n_features_cpf_,         // cpf features
      n_npf_ * n_features_npf_,         // npf features
      n_sv_ * n_features_sv_,           // vtx features
  };
  
  // init data storage
  data_.clear();
  for (const auto& len : input_sizes_) {
    data_.emplace_back(1 * len, 0);
  }

  make_inputs(features);
}





void ParticleTransformerAK4ONNXJetTagsProducer::make_inputs(
    const reco::ParticleTransformerAK4TagInfo& taginfo) {
  
  float* ptr = nullptr;
  const float* start = nullptr;
  const auto& features = taginfo.features();
  const auto& jet = taginfo.jet();
  
  // =======================
  // GLOBAL FEATURES - Match training config exactly
  // =======================
  ptr = &data_[kGlobalFeatures][0];
  start = ptr;
  
  // 1. jet_pt - Use actual jet pt from the jet object
  *ptr = jet->pt();
  
  // 2. jet_eta - Use actual jet eta from the jet object
  *(++ptr) = jet->eta();
  
  // 3. n_Cpfcand - Number of charged PF candidates
  *(++ptr) = features.c_pf_features.size();
  
  // 4. n_Npfcand - Number of neutral PF candidates  
  *(++ptr) = features.n_pf_features.size();
  
  // 5. nsv - Number of secondary vertices
  *(++ptr) = features.sv_features.size();
  
  // 6. npv - Number of primary vertices (needs event access)
  // For now, use a default value or pass via parameter
  *(++ptr) = 1;  // Default, should be extracted from event
  
  // 7-15. CSV Tagger variables
  // These need to be computed from track information or set to defaults
  // if not available from another TagInfo
  
  // 7. TagVarCSV_trackSumJetEtRatio
  float track_sum_pt = 0;
  for (const auto& cpf : features.c_pf_features) {
    track_sum_pt += cpf.pt;
  }
  *(++ptr) = jet->pt() > 0 ? track_sum_pt / jet->pt() : 0;
  
  // 8. TagVarCSV_trackSumJetDeltaR  
  float weighted_dr = 0;
  float weight_sum = 0;
  for (const auto& cpf : features.c_pf_features) {
    weighted_dr += cpf.btagPf_trackDeltaR * cpf.pt;
    weight_sum += cpf.pt;
  }
  *(++ptr) = weight_sum > 0 ? weighted_dr / weight_sum : 0;
  
  // 9. TagVarCSV_vertexCategory
  // 0: NoVertex, 1: PseudoVertex, 2: RecoVertex
  *(++ptr) = features.sv_features.size() > 0 ? 2 : 0;
  
  // 10-13. Track SIP values above charm threshold
  // These require comparing to charm mass threshold (1.5 GeV)
  // Computing approximations based on available SIP values
  float max_sip2d_val = 0, max_sip2d_sig = 0;
  float max_sip3d_val = 0, max_sip3d_sig = 0;
  
  for (const auto& cpf : features.c_pf_features) {
    // Only consider tracks with significant impact parameter
    if (std::abs(cpf.btagPf_trackSip2dSig) > 2.0) {  // Charm threshold approximation
      max_sip2d_val = std::max(max_sip2d_val, std::abs(cpf.btagPf_trackSip2dVal));
      max_sip2d_sig = std::max(max_sip2d_sig, std::abs(cpf.btagPf_trackSip2dSig));
    }
    if (std::abs(cpf.btagPf_trackSip3dSig) > 2.0) {
      max_sip3d_val = std::max(max_sip3d_val, std::abs(cpf.btagPf_trackSip3dVal));
      max_sip3d_sig = std::max(max_sip3d_sig, std::abs(cpf.btagPf_trackSip3dSig));
    }
  }
  
  *(++ptr) = max_sip2d_val;  // TagVarCSV_trackSip2dValAboveCharm
  *(++ptr) = max_sip2d_sig;  // TagVarCSV_trackSip2dSigAboveCharm
  *(++ptr) = max_sip3d_val;  // TagVarCSV_trackSip3dValAboveCharm
  *(++ptr) = max_sip3d_sig;  // TagVarCSV_trackSip3dSigAboveCharm
  
  // 14. TagVarCSV_jetNSelectedTracks
  // Count tracks passing quality cuts
  int n_selected_tracks = 0;
  for (const auto& cpf : features.c_pf_features) {
    // Basic track selection criteria
    if (cpf.pt > 1.0 &&                    // pt > 1 GeV
        cpf.numberOfPixelHits >= 2 &&      // At least 2 pixel hits
        cpf.numberOfStripHits >= 8 &&      // At least 8 strip hits
        std::abs(cpf.dz) < 0.2) {          // dz < 0.2 cm
      n_selected_tracks++;
    }
  }
  *(++ptr) = n_selected_tracks;
  
  // 15. TagVarCSV_jetNTracksEtaRel
  // Count tracks used for eta_rel calculation
  int n_tracks_eta_rel = 0;
  for (const auto& cpf : features.c_pf_features) {
    if (std::abs(cpf.btagPf_trackEtaRel) < 99) {  // Valid eta_rel value
      n_tracks_eta_rel++;
    }
  }
  *(++ptr) = n_tracks_eta_rel;
  
  // 16. had_flav_match - MC truth (0 for data, needs MC info for simulation)
  // This would need to be extracted from jet->partonFlavour() or similar
  *(++ptr) = 0;  // Default for data
  
  // 17. jet_pflavCharge - MC truth charge
  // This would need special MC information
  *(++ptr) = 0;  // Default for data
  
  assert(start + n_features_global_ - 1 == ptr);

  // c_pf candidates - now includes 4-vec at the end (31 features total)
  auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t)n_cpf_);
  for (std::size_t c_pf_n = 0; c_pf_n < max_c_pf_n; c_pf_n++) {
    const auto& c_pf_features = features.c_pf_features.at(c_pf_n);
    ptr = &data_[kChargedCandidates][offset + c_pf_n * n_features_cpf_];
    start = ptr;
    // First 27 features from yml config
    *ptr = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRel;
    *(++ptr) = c_pf_features.btagPf_trackPPar;
    *(++ptr) = c_pf_features.btagPf_trackDeltaR;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    *(++ptr) = c_pf_features.ptrel;
    *(++ptr) = c_pf_features.drminsv;
    *(++ptr) = c_pf_features.vtx_ass;
    *(++ptr) = c_pf_features.puppiw;
    *(++ptr) = c_pf_features.chi2;
    *(++ptr) = c_pf_features.quality;
    *(++ptr) = c_pf_features.pt;
    *(++ptr) = c_pf_features.charge;
    *(++ptr) = c_pf_features.dz;
    *(++ptr) = c_pf_features.btagPf_trackDecayLen;
    *(++ptr) = c_pf_features.HadFrac;
    *(++ptr) = c_pf_features.CaloFrac;
    *(++ptr) = c_pf_features.pdgID;
    *(++ptr) = c_pf_features.lostInnerHits;
    *(++ptr) = c_pf_features.numberOfPixelHits;
    *(++ptr) = c_pf_features.numberOfStripHits;
    *(++ptr) = c_pf_features.tau_signal;
    // 4-vec at the end
    *(++ptr) = c_pf_features.px;
    *(++ptr) = c_pf_features.py;
    *(++ptr) = c_pf_features.pz;
    *(++ptr) = c_pf_features.e;
    assert(start + n_features_cpf_ - 1 == ptr);
  }

  // n_pf candidates - now includes 4-vec at the end (14 features total)
  auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)n_npf_);
  for (std::size_t n_pf_n = 0; n_pf_n < max_n_pf_n; n_pf_n++) {
    const auto& n_pf_features = features.n_pf_features.at(n_pf_n);
    ptr = &data_[kNeutralCandidates][offset + n_pf_n * n_features_npf_];
    start = ptr;
    // First 10 features from yml config
    *ptr = n_pf_features.pt;
    *(++ptr) = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.etarel;
    *(++ptr) = n_pf_features.phirel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;
    *(++ptr) = n_pf_features.tau_signal;
    // 4-vec at the end
    *(++ptr) = n_pf_features.px;
    *(++ptr) = n_pf_features.py;
    *(++ptr) = n_pf_features.pz;
    *(++ptr) = n_pf_features.e;
    assert(start + n_features_npf_ - 1 == ptr);
  }

  // sv candidates - now includes 4-vec at the end (19 features total)
  auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)n_sv_);
  for (std::size_t sv_n = 0; sv_n < max_sv_n; sv_n++) {
    const auto& sv_features = features.sv_features.at(sv_n);
    ptr = &data_[kVertices][offset + sv_n * n_features_sv_];
    start = ptr;
    // First 15 features from yml config
    *ptr = sv_features.pt;
    *(++ptr) = sv_features.deltaR;
    *(++ptr) = sv_features.mass;
    *(++ptr) = sv_features.ntracks;
    *(++ptr) = sv_features.etarel;
    *(++ptr) = sv_features.phirel;
    *(++ptr) = sv_features.chi2;
    *(++ptr) = sv_features.normchi2;
    *(++ptr) = sv_features.dxy;
    *(++ptr) = sv_features.dxysig;
    *(++ptr) = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;
    *(++ptr) = sv_features.costhetasvpv;
    *(++ptr) = sv_features.enratio;
    *(++ptr) = sv_features.charge_sum;
    // 4-vec at the end
    *(++ptr) = sv_features.px;
    *(++ptr) = sv_features.py;
    *(++ptr) = sv_features.pz;
    *(++ptr) = sv_features.e;
    assert(start + n_features_sv_ - 1 == ptr);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleTransformerAK4ONNXJetTagsProducer);
