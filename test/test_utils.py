import os

from utils import download_pdf, pdf2dataset, post_process_scraped_pdf


def test_pdf_scrap():
    path = "pdf_data/inflammatory-breast-patient.pdf"
    urlprefix = "https://www.nccn.org/patients/guidelines/content/PDF/"
    d = pdf2dataset(pathes=[path])
    final_dataset = post_process_scraped_pdf(d, urlprefix=urlprefix)
    assert len(final_dataset) > 0
    assert (
        len(
            final_dataset[
                "https://www.nccn.org/patients/guidelines/content/PDF/inflammatory-breast-patient.pdf"
            ]
        )
        > 0
    )


def test_pdf_download():
    url = "https://www.nccn.org/professionals/physician_gls/pdf/breast_basic.pdf"
    path = "pdf_data/"
    filename = url.split("/")[-1]
    download_pdf(url, path)

    # Check if the file exists
    assert os.path.exists(path + "/" + filename)
    # Delete the file
    os.remove(path + "/" + filename)
