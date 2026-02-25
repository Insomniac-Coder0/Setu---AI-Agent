from typing import Dict, Any
from app.services.llm import LLMService
from app.services.google.gmail import GmailService
from app.services.google.calendar import CalendarService
from app.services.google.docs import DocsService
from app.services.google.sheets import SheetsService
from app.services.notion import NotionService
from app.services.github import GitHubService
from app.services.slack import SlackService
from app.services.discord import DiscordService
from app.core.database import get_user_credentials
import logging

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self):
        self.llm = LLMService()
    
    async def process_task(self, task_description: str, user_id: str, service: str = None) -> Dict[str, Any]:
        """
        Process user task end-to-end:
        1. Analyze with LLM to extract actions
        2. Execute each action using appropriate service
        3. Return results
        """
        
        logger.info(f"Processing task for user {user_id}: {task_description} (service: {service})")
        
        # Extract service hint from "Using <service>: ..." prefix if not explicitly provided
        service_hint = service
        clean_description = task_description
        if not service_hint and task_description.startswith("Using "):
            parts = task_description.split(":", 1)
            if len(parts) == 2:
                service_hint = parts[0].replace("Using ", "").strip()
                clean_description = parts[1].strip()
        
        # Step 1: Analyze task with service context
        analysis = self.llm.analyze_task(clean_description, service_hint=service_hint)
        
        if not analysis.get('actions'):
            return {
                "status": "failed",
                "message": "Could not understand the task",
                "results": []
            }
        
        logger.info(f"LLM analysis result: {analysis}")
        
        # Step 2: Post-LLM validation — ensure the PRIMARY action matches
        # but allow additional actions for multi-service workflows
        if service_hint and service_hint in self.llm.SERVICE_ACTION_MAP:
            expected_action = self.llm.SERVICE_ACTION_MAP[service_hint]
            if analysis['actions'] and analysis['actions'][0]['type'] != expected_action:
                logger.warning(f"LLM returned '{analysis['actions'][0]['type']}' as primary but service is '{service_hint}', overriding to '{expected_action}'")
                analysis['actions'][0]['type'] = expected_action
        
        
        results = []
        
        for action in analysis['actions']:
            try:
                result = await self._execute_action(action, user_id)
                results.append({
                    "action": action['type'],
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                logger.error(f"Action {action['type']} failed: {e}")
                results.append({
                    "action": action['type'],
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "analysis": analysis,
            "results": results
        }
    
    async def _execute_action(self, action: Dict, user_id: str) -> Dict:
        """Execute a single action"""
        
        action_type = action['type']
        params = action['parameters']
        
        # EMAIL
        if action_type == 'send_email':
            creds = await get_user_credentials(user_id, 'google')
            if not creds:
                raise Exception("Google account not connected")
            
            to = params.get('to')
            if not to:
                raise Exception("No recipient email address specified. Please include a 'to' address.")
            
            gmail = GmailService(creds['access_token'], creds.get('refresh_token'))
            return gmail.send_email(
                to=to,
                subject=params.get('subject', 'No Subject'),
                body=params.get('body', ''),
                cc=params.get('cc')
            )
        
        # CALENDAR
        elif action_type == 'create_calendar_event':
            creds = await get_user_credentials(user_id, 'google')
            if not creds:
                raise Exception("Google account not connected")
            
            summary = params.get('summary', params.get('title', 'Untitled Event'))
            start_time = params.get('start_time')
            if not start_time:
                raise Exception("No start time specified for the calendar event.")
            
            calendar = CalendarService(creds['access_token'], creds.get('refresh_token'))
            return calendar.create_event(
                summary=summary,
                start_time=start_time,
                duration_minutes=params.get('duration_minutes', 60),
                attendees=params.get('attendees'),
                description=params.get('description', '')
            )
        
        # GOOGLE DOCS
        elif action_type == 'create_google_doc':
            creds = await get_user_credentials(user_id, 'google')
            if not creds:
                raise Exception("Google account not connected")
            
            docs = DocsService(creds['access_token'], creds.get('refresh_token'))
            return docs.create_document(
                title=params.get('title', 'Untitled Document'),
                content=params.get('content', '')
            )
        
        # GOOGLE SHEETS
        elif action_type == 'create_spreadsheet':
            creds = await get_user_credentials(user_id, 'google')
            if not creds:
                raise Exception("Google account not connected")
            
            sheets = SheetsService(creds['access_token'], creds.get('refresh_token'))
            return sheets.create_spreadsheet(
                title=params.get('title', 'Untitled Spreadsheet'),
                data=params.get('data')
            )
        
        # NOTION
        elif action_type == 'create_notion_page':
            creds = await get_user_credentials(user_id, 'notion')
            if not creds:
                raise Exception("Notion account not connected")
            
            notion = NotionService(creds['access_token'])
            return notion.create_page(
                title=params['title'],
                content=params.get('content', '')
            )
        
        # GITHUB
        elif action_type == 'create_github_issue':
            creds = await get_user_credentials(user_id, 'github')
            if not creds:
                raise Exception("GitHub account not connected")
            
            repo = params.get('repo')
            if not repo:
                raise Exception("No repository specified. Use format: owner/repo")
            
            github = GitHubService(creds['access_token'])
            return github.create_issue(
                repo=repo,
                title=params.get('title', 'Untitled Issue'),
                body=params.get('body', ''),
                labels=params.get('labels')
            )
        
        # SLACK
        elif action_type == 'send_slack_message':
            creds = await get_user_credentials(user_id, 'slack')
            if not creds:
                raise Exception("Slack account not connected")
            
            slack = SlackService(creds['access_token'])
            return slack.send_message(
                channel=params.get('channel', '#general'),
                text=params.get('text', params.get('message', ''))
            )
        
        # DISCORD
        elif action_type == 'send_discord_message':
            discord = DiscordService()
            # LLM may return Slack-style params (channel/text), accept both formats
            content = params.get('content') or params.get('text') or params.get('message', '')
            channel_id = params.get('channel_id', '')
            
            if channel_id and channel_id.isdigit():
                # Use bot API with numeric channel ID
                return await discord.send_message(channel_id=channel_id, content=content)
            else:
                # No valid channel ID — use webhook fallback
                return await discord.send_webhook_message(content=content)
        
        else:
            raise Exception(f"Unknown action type: {action_type}")
