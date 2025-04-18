from collections import defaultdict
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import heapq


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-960h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0
        ):
        """
        Initialization of Wav2Vec2Decoder class
        
        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)
        
        Returns:
            str: Decoded transcript
        """
        indices = torch.argmax(logits, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i.item() for i in indices if i != self.blank_token_id]
        joined = "".join([self.vocab[i] for i in indices])
        
        return joined.replace(self.word_delimiter, " ").strip()

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring
        
        Returns:
            Union[str, List[Tuple[float, List[int]]]]: 
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        log_logits = torch.log_softmax(logits, dim=-1)
        T, V = log_logits.shape
        
        beams = [(-0.0, [])]
        heapq.heapify(beams)
        
        for t in range(T):
            canditates = []
            seen = set()
            
            while beams and len(canditates) < self.beam_width * 2:
                neg_score, seq = heapq.heappop(beams)
                score = -neg_score
                
                curr_probs = log_logits[t]
                top_probs, top_ids = curr_probs.topk(self.beam_width)
                
                for prob, idx in zip(top_probs, top_ids):
                    new_seq = seq.copy()
                    idx = idx.item()
                    
                    if idx != self.blank_token_id and (not new_seq or new_seq[-1] != idx):
                        new_seq.append(idx)
                    
                    new_score = score + prob.item()
                    seq_tuple = tuple(new_seq)
                    if seq_tuple not in seen:
                        heapq.heappush(canditates, (-new_score, new_seq))
                        seen.add(seq_tuple)
            
            beams = []
            for _ in range(min(self.beam_width, len(canditates))):
                beams.append(heapq.heappop(canditates))
        
        if return_beams:
            return [(-score, seq) for score, seq in beams]
        
        _, best_seq = beams[0]
        joined = "".join([self.vocab[id] for id in best_seq])
        
        return joined.replace(self.word_delimiter, " ").strip()

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion
        
        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
        
        Returns:
            str: Decoded transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        log_logits = torch.log_softmax(logits, dim=-1)
        T, V = log_logits.shape
        
        beams = [(-0.0, [])]
        heapq.heapify(beams)
        
        for t in range(T):
            canditates = []
            seen = set()
            
            while beams and len(canditates) < self.beam_width * 2:
                neg_score, seq = heapq.heappop(beams)
                score = -neg_score
                
                curr_probs = log_logits[t]
                top_probs, top_ids = curr_probs.topk(self.beam_width)
                
                for prob, idx in zip(top_probs, top_ids):
                    new_seq = seq.copy()
                    idx = idx.item()
                    
                    if idx != self.blank_token_id and (not new_seq or new_seq[-1] != idx):
                        new_seq.append(idx)
                    
                    joined = "".join([self.vocab[id] for id in new_seq])
                    words = joined.split(self.word_delimiter)
                    lm_text = " ".join(words).strip()
                    
                    am_score = prob.item()
                    lm_score = self.lm_model.score(lm_text) if lm_text else 0.0
                    seq_len = len(words) * self.beta if lm_text else 0.0
                    new_score = score + am_score + (self.alpha * lm_score) + seq_len
                    
                    seq_tuple = tuple(new_seq)
                    if seq_tuple not in seen:
                        heapq.heappush(canditates, (-new_score, new_seq))
                        seen.add(seq_tuple)
            
            beams = []
            for _ in range(min(self.beam_width, len(canditates))):
                beams.append(heapq.heappop(canditates))
        
        _, best_seq = beams[0]
        joined = "".join([self.vocab[id] for id in best_seq])
        
        return joined.replace(self.word_delimiter, " ").strip()

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs
        
        Args:
            beams (list): List of tuples (hypothesis, log_prob)
        
        Returns:
            str: Best rescored transcript
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        rescored_beams = []
        heapq.heapify(rescored_beams)
        
        for neg_score, seq in beams:
            score = -neg_score
            text = "".join([self.vocab[id] for id in seq])
            words = text.split(self.word_delimiter)
            lm_text = " ".join(words).strip()
            
            lm_score = self.lm_model.score(lm_text) if lm_text else 0.0
            total_score = score + (self.alpha * lm_score)
            heapq.heappush(rescored_beams, (-total_score, seq))
        
        _, best_seq = heapq.heappop(rescored_beams)
        
        joined = "".join([self.vocab[id] for id in best_seq])
        return joined.replace(self.word_delimiter, " ").strip()

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method
        
        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and 
                      "beam_lm_rescore" is a beam search with second pass LM rescoring
        
        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError("Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'.")


def test(decoder, audio_path, true_transcription):

    import Levenshtein
    import time

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    results_time = {
        'greedy': [],
        'beam': [],
        'beam_lm': [],
        'beam_lm_rescore': []
    }
    results_distance = {
        'greedy': [],
        'beam': [],
        'beam_lm': [],
        'beam_lm_rescore': []
    }

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        start_time = time.time()
        transcript = decoder.decode(audio_input, method=d_strategy)
        overall_time = time.time() - start_time
        distance = Levenshtein.distance(true_transcription, transcript.strip())
        print(f"{transcript}")
        print(f"Character-level Levenshtein distance: {distance}")

        results_time[d_strategy].append(overall_time)
        results_distance[d_strategy].append(distance)

    return results_time, results_distance

if __name__ == "__main__":

    from utils import print_statistics, aggregate_results
    
    test_samples = [
        ("examples/sample1.wav", "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE"),
        ("examples/sample2.wav", "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS"),
        ("examples/sample3.wav", "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN"),
        ("examples/sample4.wav", "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM"),
        ("examples/sample5.wav", "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES"),
        ("examples/sample6.wav", "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE"),
        ("examples/sample7.wav", "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS"),
        ("examples/sample8.wav", "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE"),
    ]

    decoder = Wav2Vec2Decoder()

    results = [test(decoder, audio_path, target) for audio_path, target in test_samples]

    results_time, results_distance = aggregate_results(results)

    print('DISTANCE COMPARISON')
    print_statistics(results_distance)
    print('TIME COMPARISON')
    print_statistics(results_time)