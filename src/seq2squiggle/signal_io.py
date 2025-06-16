#!/usr/bin/env python

"""
SLOW5 / BLOW5 writer
"""
import pyslow5
import logging
import numpy as np
import multiprocessing
import pod5
from datetime import datetime
import os
from uuid import uuid4
import uuid

logger = logging.getLogger("seq2squiggle")


def indexed_uuid(index: int) -> uuid.UUID:
    """
    Generate a UUID4-like but incrementing UUID using an index.
    """
    return uuid.UUID(f"00000000-0000-0000-0000-{index:012d}")


def get_seq_kit_and_flow_cell(profile_name: str):
    """
    Return the sequencing kit and flow cell product code based on the profile name.
    """
    mapping = {
        "rna-004": {
            "seq_kit": "sqk-rna004",
            "prom": "FLO-PRO004RA",
            "min": "FLO-MIN004RA"
        },
        "rna-002": {
            "seq_kit": "sqk-rna002",
            "prom": "FLO-PRO002",
            "min": "FLO-MIN106"
        },
        "dna-r10": {
            "seq_kit": "SQK-LSK114",
            "prom": "FLO-PRO114",
            "min": "FLO-MIN114"
        },
        "dna-r9": {
            "seq_kit": "SQK-LSK109",
            "prom": "FLO-PRO001",
            "min": "FLO-MIN110"
        }
    }

    for prefix, data in mapping.items():
        if profile_name.startswith(prefix):
            flow_cell_key = "prom" if "prom" in profile_name else "min" if "min" in profile_name else None
            if flow_cell_key is None:
                break
            return data["seq_kit"], data[flow_cell_key]

    raise ValueError(f"Unsupported profile name: {profile_name}")

class BLOW5Writer:
    """
    Export signal predictions to an slow5/blow5 file.

    Parameters
    ----------
    filename : str
        The name of the slow5 file.
    """

    def __init__(self, filename, profile, ideal_mode, profile_name):
        self.filename = filename
        self.profile: dict = profile
        self.ideal_mode = ideal_mode
        self.profile_name = profile_name
        self.signals = None
        self.median_before = float(self.profile["median_before_mean"])
        self.median_before_std = float(self.profile["median_before_std"])
        self.offset = float(self.profile["offset_mean"])
        self.offset_std = float(self.profile["offset_std"])
        self.digitisation = float(self.profile["digitisation"])
        self.signal_range = float(self.profile["range"])
        self.sample_rate = float(self.profile["sample_rate"])
        self.start_time = 0

    def save(self):
        """
        Export the spectrum identifications to the slow5 file.
        """
        if self.signals == None:
            logger.warning("SLOW5 was not exported. No signals were found")
            raise ValueError("SLOW5 was not exported. No signals were found")

        seq_kit, flow_cell_product_code = get_seq_kit_and_flow_cell(self.profile_name)
        # Check if file exists for appending mode
        f_mode = "a" if os.path.exists(self.filename) else "w"
        logger.debug(f"File mode for saving: {f_mode}")

        # To write a file, mode in Open() must be set to 'w' and when appending, 'a'
        s5 = pyslow5.Open(str(self.filename), f_mode)

        if f_mode == 'w':
            header, end_reason_labels  = s5.get_empty_header(aux=True)
            header['asic_id'] = 'asic_id_0'
            header['exp_start_time'] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            header['run_id'] = 'run_id_0'
            header['flow_cell_id'] = 'FAN00000'
            header['flow_cell_product_code'] = flow_cell_product_code
            header['experiment_type'] = 'rna' if self.profile_name.startswith("rna") else 'genomic_dna'
            header['sample_frequency'] = int(self.sample_rate)
            header['sequencing_kit'] = seq_kit

            # Remove any fields with None values from header
            header = {key: value for key, value in header.items() if value is not None}

            ret = s5.write_header(header,end_reason_labels=end_reason_labels)

        records = {}
        auxs = {}

        for idx, (read_id, signal) in enumerate(self.signals.items()):
            if len(signal) == 0:
                logger.debug("Empty signal, skipping {}".format(read_id))
                continue
            
            if self.ideal_mode:
                median_before_value = self.median_before
                offset_value = self.offset
            else:
                median_before_value = np.random.normal(self.median_before, self.median_before_std)
                offset_value = np.random.normal(self.offset, self.offset_std)
            signal = signal.cpu().numpy().astype(np.float32)
            signal_raw = np.round(
                signal * self.digitisation / self.signal_range - self.offset
            )
            signal_raw = signal_raw.astype(np.int16)

            if self.profile_name.startswith("rna"):
                signal_raw = np.ascontiguousarray(signal_raw[::-1])

            record, aux = s5.get_empty_record(aux=True)

            record["read_id"] = str(indexed_uuid(idx + 1)) #read_id
            record["read_group"] = 0
            record["digitisation"] = self.digitisation
            record["offset"] = offset_value
            record["range"] = self.signal_range
            record["sampling_rate"] = self.sample_rate
            record["len_raw_signal"] = len(signal_raw)
            record["signal"] = signal_raw

            aux["channel_number"] = "0"
            aux["median_before"] = median_before_value
            aux["read_number"] = idx
            aux["start_mux"] = 0
            aux["start_time"] = self.start_time
            self.start_time += len(signal_raw)

            records[record["read_id"]] = record
            auxs[record["read_id"]] = aux
        

        num_processes = multiprocessing.cpu_count()
        if records:
            ret = s5.write_record_batch(
                records, threads=num_processes, batchsize=500, aux=auxs
            )
        s5.close()


class POD5Writer:
    """
    Export signal predictions to a pod5 file.

    Parameters
    ----------
    filename : str
        The name of the pod5 file.
    """

    def __init__(self, filename, profile, ideal_mode, profile_name):
        self.filename = filename
        self.profile: dict = profile
        self.ideal_mode = ideal_mode
        self.profile_name = profile_name
        self.signals = None
        self.median_before = float(self.profile["median_before_mean"])
        self.median_before_std = float(self.profile["median_before_std"])
        self.offset = float(self.profile["offset_mean"])
        self.offset_std = float(self.profile["offset_std"])
        self.digitisation = float(self.profile["digitisation"])
        self.signal_range = float(self.profile["range"])
        self.sample_rate = float(self.profile["sample_rate"])
        self.start_time = 0

    def save(self):
        """
        Export the spectrum identifications to the slow5 file.
        """
        if self.signals == None:
            logger.warning("POD5 was not exported. No signals were found")
            raise ValueError("POD5 was not exported. No signals were found")

        seq_kit, flow_cell_product_code = get_seq_kit_and_flow_cell(self.profile_name)

        run_info = pod5.RunInfo(
            acquisition_id="",  # f5d5051ec9f7983c76e78543f720289d2988ce48
            acquisition_start_time=datetime.now(),
            adc_max=4095,
            adc_min=-4096,
            context_tags={},
            experiment_name="",  # choose a name
            flow_cell_id="",  # FAV99375
            flow_cell_product_code=flow_cell_product_code,  # FLO-MIN114
            protocol_name="",  # sequencing/sequencing_MIN114_DNA_e8_2_400K:FLO-MIN114:SQK-LSK114:400
            protocol_run_id="",  # a2f3daba-e515-4853-859f-79bb92079c23
            protocol_start_time=datetime.now(),
            sample_id="test",  # no_sample
            sample_rate=int(self.sample_rate),
            sequencing_kit=seq_kit,  # sqk-lsk114
            sequencer_position="",  # MN44571
            sequencer_position_type="",  # MinION Mk1B
            software="",  # MinKNOW 23.04.6 (Bream 7.5.10, Core 5.5.5, Guppy unknown)
            system_name="",  # HZI19-WW10015
            system_type="",  # Windows 10.0
            tracking_id={},
        )

        pod5_reads = []

        for idx, (read_id, signal) in enumerate(self.signals.items()):
            if len(signal) == 0:
                logger.debug("Empty signal, skipping {}".format(read_id))
                continue

            if self.ideal_mode:
                median_before_value = self.median_before
                offset_value = self.offset
            else:
                median_before_value = np.random.normal(self.median_before, self.median_before_std)
                offset_value = np.random.normal(self.offset, self.offset_std)
            signal = signal.cpu().numpy().astype(np.float32)
            signal_raw = np.round(
                signal * self.digitisation / self.signal_range - self.offset
            )
            signal_raw = signal_raw.astype(np.int16)
            
            if self.profile_name.startswith("rna"):
                signal_raw = signal_raw[::-1]

            pore = pod5.Pore(channel=123, well=3, pore_type="not_set")
            calibration = pod5.Calibration(
                offset=offset_value, scale=(self.signal_range / self.digitisation)
            )

            end_reason = pod5.EndReason(
                reason=pod5.EndReasonEnum.SIGNAL_POSITIVE, forced=False
            )

            read = pod5.Read(
                read_id=indexed_uuid(idx + 1),
                pore=pore,
                calibration=calibration,
                read_number=idx,
                start_sample=0,
                median_before=median_before_value,
                end_reason=end_reason,
                run_info=run_info,
                signal=signal_raw,
            )
            pod5_reads.append(read)

        with pod5.Writer(self.filename) as writer:
            for read in pod5_reads:
                writer.add_read(read)
