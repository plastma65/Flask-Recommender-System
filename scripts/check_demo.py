"""Check public inference without PostgreSQL or private survey records."""
import asyncio
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(ROOT / 'Project/smart-learning-backend'))
import httpx
from app.main import app

async def main():
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://demo') as c:
            r = await c.get('/api/v1/catalog/teachers')
            assert r.status_code == 200
            assert len(r.json()['data']) == 45
            for query in ['Tôi muốn học lập trình thiết bị di động.', 'Toi muon hoc lap trinh cho thiet bi di dong']:
                r = await c.post('/api/v1/recommendations/teachers', json={'query_text':query,'alpha':0.6,'top_k':5})
                assert r.status_code == 200, r.text
                assert {x['teacher_id'] for x in r.json()['data']['items']} == {6,11}
            for failed, gpa, attendance, expected in [('0','Từ 2.5 đến 3.19','>90%','PASS'),('Từ 4 môn trở lên','Từ 2.0 đến 2.49','70–90%','FAIL')]:
                r = await c.post('/api/v1/predictions/pass-fail', json={'gpa_bucket':gpa,'study_hours_bucket':'Từ 0 đến 5 giờ','failed_subjects_count':failed,'attendance_bucket':attendance})
                assert r.status_code == 200, r.text
                assert r.json()['data']['prediction_result'] == expected
    print('PASS: 45 teachers, accented/unaccented matching, PASS and FAIL demo cases.')

if __name__ == '__main__':
    asyncio.run(main())
