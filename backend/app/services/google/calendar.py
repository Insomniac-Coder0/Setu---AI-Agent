from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from dateutil import parser
import logging

logger = logging.getLogger(__name__)

class CalendarService:
    """Google Calendar operations"""
    
    def __init__(self, access_token: str, refresh_token: str = None):
        from google.auth.transport.requests import Request
        import os
        
        creds = Credentials(
            token=access_token,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        )
        
        # Auto-refresh if expired
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        
        self.service = build('calendar', 'v3', credentials=creds)
    
    def create_event(
        self,
        summary: str,
        start_time: str,  # ISO format or natural language
        duration_minutes: int = 60,
        attendees: list = None,
        description: str = ""
    ) -> dict:
        """Create calendar event"""
        try:
            # Parse start time
            if isinstance(start_time, str):
                start_dt = parser.parse(start_time)
            else:
                start_dt = start_time
            
            # If the parsed date has no year or is in the past, assume current/next year
            now = datetime.now()
            if start_dt.year < now.year:
                start_dt = start_dt.replace(year=now.year)
            
            end_dt = start_dt + timedelta(minutes=duration_minutes)
            
            logger.info(f"Creating calendar event: '{summary}' from {start_dt.isoformat()} to {end_dt.isoformat()}")
            
            # Use Asia/Kolkata timezone for Indian users
            user_timezone = 'Asia/Kolkata'
            
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_dt.isoformat(),
                    'timeZone': user_timezone,
                },
                'end': {
                    'dateTime': end_dt.isoformat(),
                    'timeZone': user_timezone,
                },
            }
            
            # Validate and filter attendees - only include valid email addresses
            valid_attendees = []
            if attendees:
                import re
                email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                valid_attendees = [email for email in attendees if email_pattern.match(str(email))]
                if valid_attendees:
                    event['attendees'] = [{'email': email} for email in valid_attendees]
                    event['conferenceData'] = {
                        'createRequest': {
                            'requestId': f"meet-{start_dt.timestamp()}",
                            'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                        }
                    }
            
            created_event = self.service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1 if valid_attendees else 0,
                sendUpdates='all' if valid_attendees else 'none'
            ).execute()
            
            logger.info(f"Event created: {created_event['id']}")
            
            return {
                "event_id": created_event['id'],
                "html_link": created_event.get('htmlLink'),
                "meet_link": created_event.get('hangoutLink'),
                "status": "created",
                "summary": summary,
                "start_time": start_dt.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            raise Exception(f"Calendar event creation failed: {str(e)}")
